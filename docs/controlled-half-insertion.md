# Controlled half-insertion before release

**The initial live trial passed the controlled insertion/release sequence. It did not establish an overall monitoring or final-rest pass.** The plug entered approximately 29 mm while held by both finger pads, then the gripper intentionally opened. The final physics sample failed `angular_rest`, and RoboBrain remained `ON_GOING` with no SUCCESS events. [Recorded metrics](half-insertion-validation-metrics.json)

## Requirement and original failure

The silver pin is 50 mm long. At least 25 mm must enter the actual socket bore **while the robot still holds the plug**. After that engagement, intentional release may let gravity finish seating. Dropping from above the opening or from its mouth does not meet the requirement, even if the plug subsequently lands fully seated.

The [original demo](../runs/recording-validation-20260924T235301Z/robobrain/runs/RoboBrain-Show_Harness-Demo) violated this sequence. After control 63, the pin tip was about **31.7 mm above the rim**, although the gripper width was still approximately 32 mm. Before control 64, following roughly 20 seconds of waiting, the gripper was already empty. The recovery `RELEASE` at control 65 followed the fall; it did not cause it. A final seating predicate alone could not distinguish this uncontrolled arrival from insertion under grasp.

## Release qualification

The controller now uses registered wrist RGB/depth plus a VLM check. The measurement is specific to this scene's orange handle, blue socket, 50 mm pin, and 32 mm handle height. With the wrist within one degree of vertical, suitable top-surface depth measurements give:

```text
insertion estimate = 82 mm − (blue top depth − orange top depth)
conservative lower bound = insertion estimate − 3 mm
```

The depth difference in this expression is in millimetres. Color components, depth consistency, and fitted surface planes must support the measurement; unsuitable or missing evidence cannot authorize release. The lower bound must reach 25 mm. In practice, the controller aims for about 28–30 mm to cover that margin.

Depth alone does not establish alignment or continued grasp. The VLM must also confirm entry into the black bore and opposed finger-pad contact from the current camera views. Fresh gripper checks cover every opening route, including recovery and internal empty-grasp reopening. Initial source-supported corrections remain distinct from destination release. The current implementation rechecks RGB-D at dispatch, invalidates stale approvals, and records the state both before and after opening.

Object poses and physical contact forces are **not controller or release-guard inputs**. The controller receives camera observations, image-derived measurements, and robot proprioception. Simulator ground truth is used separately for process audits and completion diagnostics. The sampled process audit distinguishes held engagement before release from a falling pin crossing the same depth after grasp loss.

## Continuous-physics grip stability

In MuJoCo 3.14, the unchanged contact model exhibited gradual creep during long waits: a centered grasp slipped about 4.28 mm over 120 simulated seconds; a deliberately shallow grasp lost contact after approximately 29 seconds. Gripper width remained near 32 mm until the shallow grip failed.

The plug service profile now explicitly selects:

```yaml
continuous_physics:
  cone: elliptic
  impratio: 100
  noslip_iterations: 0
```

This follows [MuJoCo's guidance for slow contact slippage](https://mujoco.readthedocs.io/en/latest/modeling.html#slow-slippage). The original Show-Harness checkout remains read-only. Scene assets, friction coefficients, actuator gains/forces, the free plug joint, and the 2 ms timestep were not changed. The standalone decision-synchronous baseline retains its original solver settings.

In the measured 120-second fixtures, relative drift fell to approximately 0.086 mm for a centered grasp and 0.035 mm for a shallow grasp; both retained loaded contacts and fell normally after explicit release. Seated physics ran at about 7.08× real time without rendering. NoSlip was rejected: its dense seated-contact solve needed additional arena memory and ran at only 0.15× real time.

The selected profile still exhibited bounded seated rocking: a five-second fixture measured about 0.38 mm maximum positional peak-to-peak motion and 0.62° maximum tilt. The original angular-rest threshold was retained, and its failures were recorded rather than reclassified as success. These fixture measurements are not end-to-end performance guarantees.

## Initial live evidence and limits

The GUI/full-recording trial used `gpt-6-sol` for control and `gpt-6-luna` for monitoring. It produced 100 logged controls and zero recorded sequence violations.

| Evidence | Result |
| --- | --- |
| Physical pin penetration immediately before opening | **29.013 mm**; required at least 25 mm |
| Grasp immediately before opening | Loaded contact on both finger pads, approximately 0.80 N normal force per finger |
| Gripper width | **31.999 → 76.610 mm**, confirming commanded opening |
| RGB-D estimate at verification | **29.016 mm**; conservative lower bound **26.016 mm** |
| Execution result | `FINISHED`; `end_reason: task_success` |
| Final physics sample | `success: false`; only `angular_rest` failed |
| RoboBrain result | `ON_GOING`; no SUCCESS events |

The execution label and final success sample disagree; neither `FINISHED` nor `end_reason` alone proves complete success. The supported conclusion is that the requested held-half-insertion-before-opening sequence occurred. [Machine-readable evidence](half-insertion-validation-metrics.json)

Review the [controlled insertion/release clip](../runs/half-insertion-validation-20260927T182859Z/robobrain/runs/20260927T182900.645562Z-48a08b93/controlled-half-insertion.mp4), [frame before qualified release](../runs/half-insertion-validation-20260927T182859Z/robobrain/runs/20260927T182900.645562Z-48a08b93/before-qualified-release.jpg), and [complete run](../runs/half-insertion-validation-20260927T182859Z/robobrain/runs/20260927T182900.645562Z-48a08b93). Correlate recordings through native frame metadata rather than matching MP4 frame numbers; see the [recording guide](combined-system-recording.md).

Earlier VLM-only calibration versions did not reliably enforce depth. The RGB-D-assisted version passed four targeted cases: hovering denied, 12 mm shallow insertion denied, 30 mm held insertion approved, and a dropped plug denied.

The initial paid trial preceded the dispatch-depth recheck, after-opening audit, and plane-normal hardening. Its archived RGB-D observation was rechecked with the final plane-fitting logic and still qualified at a 26.016 mm lower bound. The subsequent full-system run below exercised the updated code.

## Follow-up full-system run

A second run on 27 September used the updated code, including the dispatch recheck and after-opening audit. The plug was held at **29.016 mm insertion before release**, and the opening completed. All 11 videos matched their frame indexes; 77 controls, 83 controller-side model calls, and 989 monitoring calls were paired. Six monitoring requests were cancelled; no model-call errors were reported. The source-provenance check confirmed that all 119 recorded source hashes were unchanged during the run.

The executor finished, but the final physics sample still failed `angular_rest` and RoboBrain ended `ON_GOING`. This confirms the insertion/release sequence, not an overall monitoring or final-rest pass. See the [run summary](../runs/full-system-20260927T195010Z/robobrain/runs/20260927T195011.495564Z-676ec8be/run-summary.json) and [combined release clip](../runs/full-system-20260927T195010Z/robobrain/runs/20260927T195011.495564Z-676ec8be/verification/controlled-release-combined.mp4).

A subsequent recording review found **4,245 omitted 30 fps slots (15.55%)**, including 4,238 pending-buffer replacements. Approximately 910 seconds of wall time became 768.6 seconds of video. The recording throughput/timing issue is diagnosed but **not fixed by these insertion changes**; use frame metadata for actual elapsed time.

## Using the updated task

Launch the continuous service with the updated `configs/robot_mujoco_plug.yaml`, following [the runbook](combined-system-runbook.md). Review the release-verification events and trajectory in the shared run folder when assessing a trial; require the qualified held engagement to precede opening or grasp loss.

Saved RoboBrain tasks are not automatically migrated. Create the updated MuJoCo starter or explicitly review the saved task's prompts and progress criteria. The template's half-insertion stage retains the existing internal node ID `SEATED_HELD` for compatibility; its label and criteria now describe half insertion while held, not mandatory full seating before release. Final seating and retreat remain separate checks.

## Final regression checks

The final simulator suite passed **163 tests and 217 parameterized checks**. Frontend, UI, progress, simulation UI/monitor/policy, and OpenAI-adapter checks passed. The final source also checks current RGB-D again at dispatch and records release qualification only after opening completes; cancellation clears pending qualification.
