# Modified for the MuJoCo-only extraction: hardware launch paths removed.
from __future__ import annotations
from typing import Any
from interpreters.real_atomic_controller import RealAtomicController
from core.record.episode_logger import EpisodeLogger
from core.runners.real import RealEpisodeRunner
from core.v0_types import V0Config
from core.agent.stage_control import Controller, StageControlSuite
from plugins.action_ablation import ActionAblationPlugin
from plugins.action_chunk import ActionChunkPlugin
from plugins.affordance import AffordancePlugin
from plugins.config import PluginsConfig
from plugins.coords import CoordsPlugin
from plugins.deepplan import DeepPlanPlugin
from plugins.ego import EgoPlugin
from plugins.mcq import McqPlugin
from plugins.mem_text import MemTextPlugin
from plugins.proprioception import ProprioceptionPlugin
from plugins.recovery import RecoveryPlugin
from plugins.subgoal import SubgoalPlanner, SubgoalPlannerAgent, load_prompt
from plugins.wrist_frame import WristFramePlugin
from core.vlm.roles import ControllerAgent

EMPTY_GRASP_WIDTH_M = 0.005
TABLE_CONTACT_Z_M = 0.025

def recovery_empty_width_m(cfg: dict[str, Any]) -> float:
    return float(cfg.get("empty_width_m", cfg.get("grasp_min_width_m", EMPTY_GRASP_WIDTH_M)))

def recovery_open_width_m(cfg: dict[str, Any]) -> float:
    return float(cfg.get("open_width_m", cfg.get("grasp_open_width_m", 0.06)))

def high_above_table_m(cfg: dict[str, Any]) -> float:
    """Threshold X (m): above this height the controller is told to descend first, and
    (if variable_step is on) uses a coarse descent. Shared by proprioception + variable_step."""
    return float(cfg.get("high_above_table_m", 0.10))

def coarse_step_m(cfg: dict[str, Any]) -> float:
    """Coarse per-command translation (m) used by variable_step when high or lifting."""
    return float(cfg.get("coarse_step_m", 0.05))

def action_chunk_step_num(cfg: dict[str, Any]) -> int:
    """Moves committed per VLM call by action_chunk while the TARGET is far (default 3)."""
    return int(cfg.get("action_chunk_step_num", 3))

def mem_text_len(cfg: dict[str, Any]) -> int:
    """How many recent moves the mem_text 'Recent moves' line shows / the runner keeps."""
    return int(cfg.get("mem_text_len", 3))

def make_runner(
    cfg: dict[str, Any],
    prompts_cfg: dict[str, Any],
    client: Any,
    session: Any,
    controller: RealAtomicController,
    logger: EpisodeLogger,
    debug: bool,
    viewer: Any = None,
    video_ref_plugin: Any = None,
    dagger_plugin: Any = None,
) -> RealEpisodeRunner:
    v0_config = V0Config.from_dict(cfg.get("v0", {}))
    common_context = prompts_cfg["common_context"]
    plugins = PluginsConfig.from_config(cfg)
    # DeepPlan: when enabled, the planner prompt gains the <REASON>-pivot rules and the
    # runner resolves a reached pivot on the live scene. Disabled -> None: no augmentation
    # and an unreachable pivot path, so the planner build below is byte-identical to today.
    dp_enabled = plugins.enabled("deepplan", default=False)
    deepplan_plugin = (
        DeepPlanPlugin(enabled=True, client=client, common_context=common_context)
        if dp_enabled
        else None
    )
    # Subgoal tool: build the planner only when enabled; otherwise the runner falls back to a
    # single whole-task stage (planner=None). When DeepPlan is on, the REASON-pivot rules are
    # appended to the planner prompt via the existing prompt_template override (the file on
    # disk is never edited); off -> prompt_template=None -> the untouched co-located prompt.
    if plugins.enabled("subgoal"):
        planner_template = None
        if dp_enabled:
            planner_template = (
                load_prompt() + "\n\n" + deepplan_plugin.render_planner_addendum()
            )
        planner = SubgoalPlanner(
            SubgoalPlannerAgent(
                client=client,
                common_context=common_context,
                prompt_template=planner_template,
                # The reference-demo brief rides on the agent, so the initial plan
                # AND every replan replicate the same demonstration ("" when off).
                video_ref_block=(
                    video_ref_plugin.render_prompt() if video_ref_plugin is not None else ""
                ),
                # Long plans (many pieces x 6 stages) on pretty-printing backends
                # (gemini) overflow small budgets and silently degrade the plan.
                max_tokens=int(cfg.get("planner_max_tokens", 4096)),
            )
        )
    else:
        planner = None
    vlm_cfg = cfg.get("vlm", {})
    cot_mode = bool(vlm_cfg.get("reasoning_cot"))
    backend_prompt = prompts_cfg.get(f"controller_{vlm_cfg.get('backend')}_prompt") if cot_mode else None
    controller_prompt = backend_prompt or prompts_cfg["controller_prompt"]
    controller_prompt = CoordsPlugin(plugins.enabled("coords", default=False)).apply(controller_prompt)
    # Optional wrist motion frame (motion_frame: wrist): MV_* follow the gripper's
    # heading, so the front-view direction lines are rewritten to heading-relative
    # judgments. Default base -> untouched (the AgentView-centric convention).
    wrist_frame = str(cfg.get("motion_frame", "base")).strip().lower() == "wrist"
    if wrist_frame and plugins.enabled("coords", default=False):
        print(
            "[run-real] WARNING: plugins.coords writes base-axis directions, but "
            "motion_frame: wrist executes MV_* along the gripper heading -- the "
            "coords scheme will not match the executed motion."
        )
    controller_prompt = WristFramePlugin(wrist_frame).apply(controller_prompt)
    # Egocentric scene camera (e.g. the Piper left-arm rig): the FWD/BACK <-> image
    # depth mapping is inverted vs the Franka convention baked into controller.txt.
    # Driven by the hardware `is_ego` flag; a non-ego rig leaves the prompt untouched.
    controller_prompt = EgoPlugin(bool(cfg.get("is_ego", False))).apply(controller_prompt)
    # Action-type ablation (action_ablation_mode: off|bare|letters|letters_blind):
    # the no-explanation settings replace the DIRECTION section at template level;
    # the letters modes also swap the answer alphabet to opaque ACT_* symbols
    # (agent-side; blind adds the two-frame review + self-written table).
    action_ablation_plugin = ActionAblationPlugin(
        mode=str(cfg.get("action_ablation_mode", "off") or "off"),
        include_rotate=bool(getattr(controller.rotation_plugin, "enabled", False)),
    )
    controller_prompt = action_ablation_plugin.apply(controller_prompt)
    if action_ablation_plugin.enabled:
        print(f"[run-real] action-type ablation: {action_ablation_plugin.mode}")
        if action_ablation_plugin.answer_protocol and plugins.enabled("mcq", default=False):
            print("[run-real] note: plugins.mcq is ignored while the ablation owns the answer alphabet.")
    gripper_color = str(cfg.get("gripper_color", "black"))
    recovery_plugin = RecoveryPlugin(
        enabled=plugins.enabled("recovery"),
        empty_width_m=recovery_empty_width_m(cfg),
        open_width_m=recovery_open_width_m(cfg),
    )
    # Use the simulator's tabletop height when available. Hardware retains its
    # calibrated contact reference; the controller's safety floor is independent.
    table_height_m = getattr(
        session, "table_height_m",
        controller.z_floor_m if controller.z_floor_m is not None else TABLE_CONTACT_Z_M,
    )
    # action_chunk (runner-level decision cadence): repeat a move step_num times while the
    # TARGET is far. Shared with the controller agent so the WRIST marker is rendered/parsed
    # whenever this OR variable_step is enabled.
    action_chunk_plugin = ActionChunkPlugin(
        enabled=plugins.enabled("action_chunk", default=False),
        step_num=action_chunk_step_num(cfg),
    )
    # mem_text owns the "Recent moves" line; its length (max_recent) is shared with the
    # runner, which retains exactly that many moves.
    mem_text_plugin = MemTextPlugin(plugins.enabled("mem_text"), max_recent=mem_text_len(cfg))
    # Affordance dots: the runner grounds each stage's contact point and premarks it on
    # the AgentView; the controller agent rewrites AFFORD to bind that dot.
    affordance_plugin = AffordancePlugin(
        enabled=plugins.enabled("affordance", default=False),
        client=client,
        verify_rounds=int(cfg.get("affordance_verify_rounds", 1)),
        # The single-arm prompt names the scene image "AgentView"; describe it to the
        # pointer as what it physically is on this rig (an external upper camera).
        view_desc="the AgentView: an external camera's upper view of the robot workspace",
        # affordance_wrist_track: false -> front-only dots (no wrist VLM calls).
        wrist_tracking=bool(cfg.get("affordance_wrist_track", True)),
    )
    vstep = controller.variable_step_plugin
    cartesian_actions = getattr(controller, "cartesian_actions", None)
    if cartesian_actions is not None and (
        plugins.enabled("mcq", default=False) or action_ablation_plugin.enabled
        or action_chunk_plugin.enabled or getattr(controller.rotation_plugin, "enabled", False)
    ):
        raise ValueError("cartesian_motion requires the single-action protocol; disable "
                         "mcq, action_ablation, action_chunk, and the legacy rotation plugin")
    if action_chunk_plugin.enabled and vstep is not None and vstep.enabled:
        # Both far-strategies stack: while the TARGET is far, each VLM decision commits
        # step_num coarse moves open-loop (no re-observation). Surface the resulting reach.
        print(
            f"[run-real] action_chunk + variable_step both on: when the TARGET is far, each "
            f"decision commits {action_chunk_plugin.step_num} x {vstep.coarse_step_m:.3f} m = up "
            f"to {action_chunk_plugin.step_num * vstep.coarse_step_m:.3f} m open-loop before re-checking."
        )
    controls = StageControlSuite(
        controller=Controller(
            ControllerAgent(
                client=client,
                prompt_template=controller_prompt,
                common_context=common_context,
                cot_mode=cot_mode,
                gripper_color=gripper_color,
                proprio_plugin=ProprioceptionPlugin(
                    plugins.enabled("proprioception"),
                    high_above_table_m=high_above_table_m(cfg),
                    # Surface the real per-step distances: the controller's fine step, and the
                    # coarse step only when variable_step is on (else every step is fine).
                    fine_step_m=controller.step_m,
                    coarse_step_m=(vstep.coarse_step_m if (vstep is not None and vstep.enabled) else None),
                    large_step_m=(getattr(vstep, "large_step_m", None) if (vstep is not None and vstep.enabled) else None),
                    step_sizes_text=(f"Bare MV_* moves {controller.step_m * 1000:g} mm; "
                                     "sized tokens use the SMALL, MEDIUM, or LARGE sizes below"
                                     if cartesian_actions is not None else None),
                ),
                mcq_plugin=McqPlugin(plugins.enabled("mcq", default=False)),
                mem_text_plugin=mem_text_plugin,
                # Same instance the controller uses, so (when enabled) it renders the
                # {variable_step} prompt and parses the VLM's wrist-visibility judgment.
                variable_step_plugin=controller.variable_step_plugin,
                action_chunk_plugin=action_chunk_plugin,
                # Same instance the controller uses for yaw compensation, so the prompt block
                # + offered ROTATE tokens stay in lockstep with the executor.
                rotation_plugin=controller.rotation_plugin,
                # Same instance the runner grounds/annotates with, so the rewritten
                # AFFORD field always names a dot that is actually drawn on the frame.
                affordance_plugin=affordance_plugin,
                action_ablation_plugin=action_ablation_plugin,
                table_height_m=table_height_m,
                cartesian_actions=cartesian_actions,
            )
        ),
    )
    return RealEpisodeRunner(
        session=session,
        controller=controller,
        planner=planner,
        controls=controls,
        logger=logger,
        config=v0_config,
        task=str(cfg["task"]),
        gripper_color=gripper_color,
        max_steps=int(cfg["max_steps"]),
        loop_period_s=float(cfg.get("loop_period_s", 0.0)),
        use_wrist_image=bool(cfg.get("use_wrist_image", True)),
        debug=debug,
        recovery_plugin=recovery_plugin,
        action_chunk_plugin=action_chunk_plugin,
        deepplan_plugin=deepplan_plugin,
        affordance_plugin=affordance_plugin,
        dagger_plugin=dagger_plugin,
        # Blind-mode review frames + action_table.json persistence; off -> inert.
        action_ablation_plugin=action_ablation_plugin,
        recent_moves_max=mem_text_plugin.max_recent,
        visual_history_steps=int(cfg.get("visual_history_steps", 0)),
        viewer=viewer,
    )
