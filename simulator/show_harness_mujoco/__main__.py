"""Launch the independent continuous-physics service on loopback port 8766."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-config", default=str(root / "configs/robot_mujoco_plug.yaml"))
    parser.add_argument("--vlm-backend", default="openai")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--record-global", action="store_true")
    parser.add_argument("--run-root", type=Path, help="Shared artifact root (default: working directory/.robobrain/runs)")
    parser.add_argument("--no-controller", action="store_true", help="Exercise physics/cameras/control protocol without provider requests")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    config = Path(args.robot_config)
    if not config.is_file() and (root / config).is_file():
        config = root / config
    if not config.is_file():
        parser.error(f"Robot configuration does not exist: {config}")
    # The baseline source tree retains its original top-level modules.
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from core.sim.service_runtime import ServiceRuntime
    from .service import create_app
    import uvicorn
    runtime = ServiceRuntime(config_path=config.resolve(), vlm_backend=args.vlm_backend,
                             gui=args.gui, record=args.record, record_global=args.record_global,
                             no_controller=args.no_controller, run_root=args.run_root)
    uvicorn.run(create_app(runtime), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
