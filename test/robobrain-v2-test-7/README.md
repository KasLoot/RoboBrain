# RoboBrain experiment 7

First standalone three-specialist implementation of the shared `Specialist(system_prompt, config)` workflow. A single verification round collects ten samples for each enabled specialist and waits for all 30 responses. This version's initial rotation prompt was rejected during image screening because it confused three released, seated adapters with rotation failures. See [results](RESULTS.md).

```sh
uv run --script test/robobrain-v2-test-7/check.py
uv run --script test/robobrain-v2-test-7/probe.py \
  --videos video/insert_plug-full_model-aug-28-105trail-1x \
  --output test/robobrain-v2-test-7/my-probe
```

`run.py --videos PATH --output NEW_DIRECTORY` runs streaming episodes; `--episodes 9 82` selects clips. It uses only code, configuration, prompts and references inside this folder. Controller handoffs are simulated log events; no robot API is called. Production code was not changed.
