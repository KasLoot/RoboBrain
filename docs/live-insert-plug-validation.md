# Live insert-plug validation

Runs started 2026-09-24 UTC; report finalized 2026-09-25 (Europe/London).

**Result: the integration workflow was demonstrated.** The integrated applications ran with the requested OpenAI models, and the robot reached the simulator's successful final state. Start/stop/resume, independent monitoring, camera control and recording worked in these runs. The user subsequently clarified that post-success status variation is acceptable at the reset boundary and that insertion-quality optimization is outside scope. The observations below are preserved; this historical run does not validate the newer [per-run recording layout](combined-system-recording.md).

## Requested configuration

- Robot controller: **OpenAI `gpt-6-sol`**, medium reasoning, original plug task and action sizes, continuous physics, GUI, camera recording and Full HD global recording.
- RoboBrain: **OpenAI `gpt-6-luna`**. The live calibrated profile used low reasoning, a 1,024-token output budget, two current camera views, labeled initial/completed reference images, two active outcome specialists, and an independent progress observer.
- Verification remained enabled: ten sampled observations per specialist, requiring more than five positive votes. Automatic robot stopping was disabled for observation of the complete attempt.
- The original Show-Harness checkout remained read-only. All code, reference images, experiments and recordings were written within RoboBrain.

This was an iterative validation: the monitoring profile and reference wording were calibrated during the requested-model run. The final calibrated profile has not yet passed a separate uninterrupted full trial. The stationary-scene variation remains evidence about model behavior, rather than an acceptance blocker for the clarified recording scope.

The official OpenAI adapter uses environment-based authentication, structured JSON, `max_completion_tokens`, and the selected reasoning effort. It omits the local-vLLM-only `chat_template_kwargs`; sampling temperature is omitted with low reasoning. See [GPT-6 Luna's documented capabilities](https://developers.openai.com/api/docs/models/gpt-6-luna) and [OpenAI's parameter migration guidance](https://developers.openai.com/api/docs/guides/latest-model#update-api-and-model-parameters).

## Results

| Area | Result | Evidence |
| --- | --- | --- |
| Simulator final state | Pass | `task_success`, 72 logged control steps, all eight physical criteria passed |
| Task duration | About 15.0 minutes | Includes a deliberate pause/resume check |
| Final TCP clearance | Pass | 86.85 mm above plug top; required more than 50 mm |
| Gripper release | Pass | Final width 77.72 mm; required more than 60 mm |
| Pin alignment | Pass | Tip offset 0.94 mm; required less than 4 mm |
| Seating and rest | Pass | Insertion depth, shaft fit, uprightness, linear rest and angular rest all passed |
| Stop/resume | Pass after repair | Stop applied in 0.52 ms on the service clock; physics advanced 3.04 seconds during the held interval; attempt identity and gripper control were preserved |
| Resume plan/cursor | Pass after repair | Same plan retained; first resumed step remained MOVE with `MV_RIGHT_LARGE`, without replaying completed GRASP/LIFT stages |
| Dynamic cameras | Pass | Inspection viewpoint changed during monitoring; frame revisions changed while the robot attempt continued |
| API/stream health | Pass | No HTTP errors in the requested-model monitoring segments; no simulator, renderer or recorder error at completion |
| Visual success detection | Detected; later cleared | Luna verified SUCCESS with 7/10 positive votes, then subsequent normal observations cleared it to ON_GOING on the unchanged completed scene |
| Progress stability | Observed variation | Stages were observed, but the final scene alternated between RETREATED, RELEASED and uncertainty |
| Manipulation quality | Outside acceptance scope | A `lost_grasp` recovery release occurred at step 71 before the final-state success predicate ended the episode |

The robot's final-state success is not proof of a clean planned insertion/release/retreat sequence. Around step 70, the recorded insertion depth changed substantially; the following step used the lost-grasp recovery override. The slip specialist was disabled in the starter, so this run does not validate reliable slip detection. The table-top drop detector correctly did not equate a plug seated in the socket with a table-top drop.

## Issues found and changes made

1. **Resume repeated a lift milestone.** The first live attempt replanned after a pause and requested another 50 mm lift. The service now retains a generation-guarded per-attempt plan, stage cursor, step budget, and stage-entry measured TCP pose. Controller context reports cumulative robot-stage travel and still requires visual confirmation of object motion. A second live pause/resume test retained the plan and cursor. Interrupted-run audit finalization was also hardened.
2. **Open fingers were mislabeled GRASPED.** The original “straddle” wording was ambiguous. GRASPED now requires visible closure/contact on both sides, with UNKNOWN for obscured contact. Captured open-gripper cases were rechecked.
3. **The initial Ollama fallback exceeded provider concurrency.** Before the requested monitoring model was selected, an Ollama verification burst produced HTTP 429. Outcome and progress requests now share a concurrency budget; rate-limited normal samples back off before taking a fresh observation. OpenAI monitoring had no HTTP errors in the measured run.
4. **Luna confused scene roles and retreat scale.** Labeled reference images identify the gray source stand and blue destination socket. Low reasoning improved the calibration examples. The completed reference has about 117 mm clearance; the prompt now explicitly states that this is an example, not the 50 mm minimum.

## Observed post-success variation and acceptance scope

In the requested-model attempt, a verification round returned seven positive success votes and zero drop votes. The current continuous-monitoring policy clears a held result when its confirming specialist stops voting positive. Later false-negative clearance/seating judgments therefore changed the display back to ON_GOING even though the simulator's completed state persisted.

A subsequent stationary-scene check, with the reference/minimum distinction clarified, again verified SUCCESS (8/10 positives) and subsequently cleared it. This is a measured stability issue, not a transport error. During that check a camera change also correctly cancelled a later verification round; that cancellation should not be counted as a perception failure.

For the main calibrated live segment, outcome-request latency was approximately **2.96 s median** and **5.31 s at the 95th percentile**. Its successful verification round completed about 8.7 seconds after triggering. These are measurements from one run, not real-time guarantees or accuracy estimates.

The user accepts this post-success variation for a production workflow that resets after completion; implementing automatic success-triggered reset remains future work. The current recording task does not require a stability fix or insertion-quality optimization. These measurements still do not establish general detector accuracy or production reliability.

## Artifacts

- [Physical success summary](../runs/live-plug-luna-20260924T222454Z/simulator/6a29c65ef9cc42df925d750f51179764/generation_3/openai/0925/task_0/06-33-29/summary.json)
- [Calibrated RoboBrain report](../runs/live-plug-luna-20260924T222454Z/robobrain/runs/task_9911bfde0097/20260924-223341-774536/output-20260924-223341-774536.json)
- [Annotated monitoring video](../runs/live-plug-luna-20260924T222454Z/robobrain/runs/task_9911bfde0097/20260924-223341-774536/output-20260924-223341-774536.MOV)
- [Stationary final-scene recheck](../runs/live-plug-luna-20260924T222454Z/luna-final-scene-recheck-result.json)
- [Live stop/resume checkpoint evidence](../runs/live-plug-luna-20260924T222454Z/checkpoint-live-check/result.json)
- [Dynamic inspection-camera update](../runs/live-plug-luna-20260924T222454Z/socket-inspection-view.json)
- [Final raw overview](../runs/live-plug-luna-20260924T222454Z/final-camera1.png) and [inspection view](../runs/live-plug-luna-20260924T222454Z/final-camera2.png)
- [Machine-readable metrics](live-validation-metrics.json)
- [Complete requested-model run directory](../runs/live-plug-luna-20260924T222454Z)

The earlier run is retained separately for comparison and reference provenance. It used `gpt-6-sol` control and an Ollama monitoring fallback before the user selected `gpt-6-luna`; it is not evidence for Luna's live accuracy. This was not a controlled comparison between models.

## Regression and integrity checks

After the changes, the simulator suite passed **93 tests and 184 subtests**. OpenAI request/authentication and shared-concurrency checks passed, along with the existing UI, Builder, endpoint, progress, frontend and simulation integration checks. The live runs exercised the native MuJoCo GUI and real model calls, extending the earlier headless/mock validation.

All 175 copied upstream file hashes still match Show-Harness, and its Git working tree is clean. The validation services were stopped after recordings finalized. Launch instructions and the current model profile are in [the runbook](combined-system-runbook.md).
