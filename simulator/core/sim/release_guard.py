"""Visual qualification of plug release, using camera evidence only.

The guard never reads simulator object poses, contacts, success predicates, or
scene geometry. The 50 mm shaft / 25 mm minimum are task specifications. A positive
classification authorizes an attempted release; it is not proof of physical depth.
"""
from __future__ import annotations

from copy import deepcopy
import json
from io import BytesIO
import numpy as np
from uuid import uuid4

from core.record.images import frame_fingerprint, vlm_camera_views
from core.sim.mujoco_depth import depth_prompt
from core.sim.visual_insertion import estimate_insertion

VERDICTS = ("HALF_INSERTED_HELD", "SUPPORTED_SOURCE_REGRASP", "UNSAFE", "UNCERTAIN")
RELEASE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": list(VERDICTS)},
        "evidence": {"type": "string"},
    },
    "required": ["verdict", "evidence"],
    "additionalProperties": False,
}

PROMPT = """Check whether it is safe to OPEN the robot gripper for this plug task.
Judge only the CURRENT four simultaneous RGB views and their matching Wrist depth
text. Do not choose a robot action or infer an object's position from a stage name.

Views: A Wrist, B Front, C Right Side, D Angled Wrist. A and D move with the hand.
B and C are independent orthographic views tilted downward. D exposes the actual
black bore rim and silver shaft. Image overlap or tip/hole pixel coincidence in
an oblique view does not prove entry. Check actual shaft/rim geometry across views.

Task dimensions: the silver pin is 50 mm long and 20 mm in diameter. The orange
handle is 32 mm tall. These are fixed dimensions, not a measurement of this scene.
At least 25 mm (half) must ALREADY be INSIDE the actual black bore of the blue socket
WHILE the handle remains visibly held between opposed finger pads.

MEASURE THE EXPOSED SHAFT, NOT THE HIDDEN TIP:
Once the shaft visibly enters the bore, its buried tip is EXPECTED to be invisible.
Do not require seeing the tip or the buried 25 mm through the opaque socket. Identify
the shaft/handle junction at the UNDERSIDE of the orange handle and the actual bore
entry plane where the shaft passes through the rim. Estimate the shaft length ABOVE
that plane along its axis. Then use: inserted length = 50 mm - exposed length.
At most 25 mm exposed qualifies; a longer exposed segment does not. For example,
20 mm exposed means about 30 mm inserted and qualifies IF it is still held and in
the bore; 35 mm exposed means only about 15 mm inserted and does not qualify. These
are arithmetic examples, not descriptions of the current images. A short silver
stub between the orange handle and rim can therefore be POSITIVE depth evidence,
not evidence of shallow insertion. Full seating or handle-to-socket contact is NOT
required: a visible gap of up to about 25 mm along the pin can still qualify.

The IMAGE-DERIVED HEIGHT CHECK below measures the visible orange handle TOP and
blue socket TOP from registered Wrist RGB and raw depth. With the calibrated camera
within one degree of vertical, top gap = blue depth - orange depth; insertion
estimate = (50 mm pin + 32 mm handle) - top gap. Its lower bound subtracts a 3 mm
uncertainty margin. These are observed image/sensor measurements, not object poses,
contacts, or simulator ground-truth state. Use this measured lower bound instead
of estimating millimetres from a projected silhouette, the 2D visible gap, or
apparent shaft width. HALF_INSERTED_HELD requires the check's qualified=true and
lower_bound_m >= 0.025. If the bound is below 25 mm, return UNSAFE; if the check is
uncertain/unavailable, do not qualify destination release. Source-supported initial
regrasp can still be assessed visually when the caller permits that separate case.

A passing surface-height check is NOT evidence of actual bore entry or holding:
an off-axis pin next to the socket can have the same top height. Use B/C/D to verify
that the shaft passes through the actual black bore, with opposed pads still on
the handle. D's perspective and handle occlusion must not substitute for the metric
check. The tip inside the opaque bore need not be visible. An occluded view is
uninformative by itself; rely on views that show the relevant alignment/contact.

Verify that the shaft actually passes through the black bore, rather than hovering
in front of it or resting on the rim. Confirm opposed pad contact on the handle:
closed empty fingers and a plug already dropped into the socket do NOT qualify.
Do not count shaft hidden by fingers, the handle, or the outside of the socket as
buried. TCP descent, commanded actions, gripper command state, and an earlier good
image are insufficient. If identifiable geometry supports less than 25 mm insertion,
return UNSAFE. If geometry is obscured or the estimate genuinely straddles the limit,
return UNCERTAIN. Do not reject a clearly short exposed shaft merely because the
inserted part and its tip are hidden inside the bore.

Choose exactly one verdict:
HALF_INSERTED_HELD: current views clearly support at least 25 mm of this 50 mm pin
inside the actual bore while the handle remains visibly held. Release may then let
gravity finish seating. This is the only destination-release qualification.
SUPPORTED_SOURCE_REGRASP: the plug is visibly resting on its original GRAY source
stand/surface, so opening merely permits a new source grasp. This is allowed only
when the caller explicitly permits it. The destination socket or rim is NOT the
source support, and a stage name alone does not establish source support.
UNSAFE: visible evidence contradicts the necessary condition (including <25 mm,
hovering, rim rest, slip, empty grip, or an already dropped plug).
UNCERTAIN: occlusion, contradictory views, or inadequate depth evidence prevents a
clear decision. Sparse depth only describes visible sampled surfaces; missing
samples and occlusion do not prove insertion. Request no hidden state or coordinates.

Return exactly JSON with verdict and evidence. Give a short visible-evidence
explanation citing the supplied image-derived depth bound plus visible bore entry
and finger contact, or why the source stand clearly supports regrasp. Do not invent
a different millimetre estimate from the silhouettes. Do not provide a plan or reasoning trace.
"""


class ReleaseNotQualified(RuntimeError):
    execution_cancelled = False

    def __init__(self, verdict, evidence, *, check_id=None, observation=None, error_type=None):
        self.details = {"verdict": verdict, "evidence": str(evidence), "check_id": check_id,
                        "observation": observation, "error_type": error_type}
        super().__init__(f"Release not qualified ({verdict}): {evidence}")


class VisualReleaseGuard:
    def __init__(self, client, artifacts=None, *, max_tokens=None):
        self.client, self.artifacts = client, artifacts
        self.max_tokens = int(max_tokens if max_tokens is not None else getattr(client, "max_tokens", 4096))
        if self.max_tokens < 1:
            raise ValueError("Release-verification max_tokens must be positive")

    def __call__(self, session, *, stage, source, carry_established, observation=None):
        check_id = uuid4().hex
        artifacts = self.artifacts or getattr(session, "artifacts", None)
        identity = None
        measurement = None
        previous_context = deepcopy(getattr(session, "audit_context", {}))
        context = getattr(session, "audit_context", None)
        source_allowed = str(stage).upper() == "GRASP" and not carry_established

        def log(event, **fields):
            if artifacts is not None:
                artifacts.append("release-verification", {"event": event, "check_id": check_id,
                    "stage": stage, "source": source, "carry_established": bool(carry_established),
                    "observation": identity, "measurement": measurement, **fields})
        try:
            obs = observation if observation is not None else session.get_observation()
            identity = deepcopy(obs.get("simulation_snapshot"))
            measurement = estimate_insertion(obs)
            required = ("server_instance_id", "episode_id", "attempt_id", "execution_generation", "state_seq")
            if not isinstance(identity, dict) or any(key not in identity for key in required):
                raise ReleaseNotQualified("UNCERTAIN", "Fresh identified simulation observation is required", check_id=check_id)
            views = vlm_camera_views(obs, obs["agentview"], obs.get("wrist"))
            if [name for name, _ in views] != ["Wrist", "Front", "Side", "Angled Wrist"]:
                raise ReleaseNotQualified("UNCERTAIN", "All four current plug camera views are required", check_id=check_id, observation=identity)
            calibration = obs.get("wrist_depth_calibration")
            text = obs.get("wrist_depth_text")
            if not isinstance(calibration, dict) or not isinstance(text, str) or not text or len(text) > 262144:
                raise ReleaseNotQualified("UNCERTAIN", "Matching calibrated Wrist depth text is required", check_id=check_id, observation=identity)
            depth = json.loads(text)
            height, width = views[0][1].shape[:2]
            if (depth.get("source_width_px"), depth.get("source_height_px")) != (width, height):
                raise ReleaseNotQualified("UNCERTAIN", "Wrist depth registration does not match its RGB view", check_id=check_id, observation=identity)
            prompt = (PROMPT + "\nSource-supported initial regrasp permitted: " + ("YES" if source_allowed else "NO")
                      + "\nIMAGE-DERIVED HEIGHT CHECK:\n" + json.dumps(measurement, allow_nan=False)
                      + "\n" + depth_prompt(calibration) + "\nCURRENT WRIST_DEPTH_MM:\n" + text)
            if isinstance(context, dict):
                context.clear()
                context.update({**previous_context, "role": "release_verification", "phase": "release_verification",
                                "stage": stage, "source": source, "carry_established": bool(carry_established),
                                "observation": identity, "release_check_id": check_id})
            depth_asset = None
            if artifacts is not None and obs.get("wrist_depth_m") is not None:
                buffer = BytesIO()
                np.save(buffer, np.asarray(obs["wrist_depth_m"]), allow_pickle=False)
                depth_asset = artifacts.save_bytes(f"release-verification/{check_id}/wrist-depth.npy", buffer.getvalue())
            log("release_verification_started", model=getattr(self.client, "model", None), depth_asset=depth_asset,
                views=[{"camera": name, **frame_fingerprint(image)} for name, image in views])
            response = self.client.complete_json(prompt, views[0][1], wrist_image=[image for _, image in views[1:]],
                schema=deepcopy(RELEASE_SCHEMA), max_tokens=self.max_tokens, temperature=0.,
                agentview_label="A: CURRENT Wrist RGB", wrist_label="CURRENT additional views in order: B Front, C Right Side, D Angled Wrist")
            payload = getattr(response, "payload", {})
            decision = payload.get("json") if isinstance(payload, dict) else None
            if decision is None:
                decision = json.loads(response.raw_text)
            if (not isinstance(decision, dict) or set(decision) != {"verdict", "evidence"}
                    or decision.get("verdict") not in VERDICTS or not isinstance(decision.get("evidence"), str)
                    or not decision["evidence"].strip() or len(decision["evidence"]) > 8192):
                raise ReleaseNotQualified("UNCERTAIN", "Invalid release-verification response", check_id=check_id, observation=identity)
            approved = (decision["verdict"] == "HALF_INSERTED_HELD" and measurement["qualified"]) or (
                decision["verdict"] == "SUPPORTED_SOURCE_REGRASP" and source_allowed)
            log("release_verification_finished", approved=approved, **decision)
            if not approved:
                evidence = decision["evidence"]
                verdict = decision["verdict"]
                if verdict == "SUPPORTED_SOURCE_REGRASP":
                    evidence = "Source regrasp exemption is unavailable after carry or outside GRASP. " + evidence
                elif verdict == "HALF_INSERTED_HELD":
                    verdict = "UNSAFE" if measurement["quality"] == "measured" else "UNCERTAIN"
                    evidence = "Image-derived depth requirement failed: " + measurement["reason"] + ". Model claim: " + evidence
                raise ReleaseNotQualified(verdict, evidence, check_id=check_id, observation=identity)
            return {**decision, "check_id": check_id, "observation": identity, "measurement": measurement}
        except ReleaseNotQualified as exc:
            exc.details["measurement"] = measurement
            log("release_verification_denied", **exc.details)
            raise
        except Exception as exc:
            if getattr(exc, "execution_cancelled", False) or "cancel" in type(exc).__name__.lower():
                log("release_verification_cancelled", error_type=type(exc).__name__)
                raise
            # Provider/audit failure can never become permission to open the gripper.
            failure = ReleaseNotQualified("UNCERTAIN", "Visual release verification failed", check_id=check_id,
                                          observation=identity, error_type=type(exc).__name__)
            failure.details["measurement"] = measurement
            log("release_verification_error", **failure.details)
            raise failure from exc
        finally:
            if isinstance(context, dict):
                context.clear()
                context.update(previous_context)
