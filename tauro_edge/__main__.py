#!/usr/bin/env python
"""Main entry point for tauro_edge module.

This allows running different edge servers:
- python -m tauro_edge robot    # Run robot server
- python -m tauro_edge sensor   # Run sensor server
- python -m tauro_edge simulator # Run simulator server
"""

import argparse
import logging
import sys
from pathlib import Path

from tauro_common.constants import DEFAULT_GRPC_HOST, DEFAULT_GRPC_PORT


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Tauro Edge Server")

    # Add subcommands
    subparsers = parser.add_subparsers(dest="command", help="Server type to run")

    # Robot server
    robot_parser = subparsers.add_parser("robot", help="Run robot control server")
    robot_parser.add_argument("--host", default=DEFAULT_GRPC_HOST, help="Host to bind to")
    robot_parser.add_argument("--port", type=int, default=DEFAULT_GRPC_PORT, help="Port to bind to")
    robot_parser.add_argument("--log-level", default="INFO", help="Logging level")

    # Sensor server
    sensor_parser = subparsers.add_parser("sensor", help="Run sensor/camera server")
    sensor_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    sensor_parser.add_argument("--port", type=int, default=50052, help="Port to bind to")
    sensor_parser.add_argument("--config", type=Path, help="Path to sensor configuration")
    sensor_parser.add_argument("--log-level", default="INFO", help="Logging level")

    # Simulator server
    sim_parser = subparsers.add_parser("simulator", help="Run MuJoCo simulator server")
    sim_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    sim_parser.add_argument("--port", type=int, default=50053, help="Port to bind to")
    sim_parser.add_argument("--config", type=Path, help="Path to simulator configuration")
    sim_parser.add_argument("--log-level", default="INFO", help="Logging level")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger(__name__)

    try:
        if args.command == "robot":
            from tauro_edge.server.robot_server import serve

            logger.info(f"Starting robot server on {args.host}:{args.port}")
            serve(args.host, args.port)

        elif args.command == "sensor":
            from tauro_edge.server.sensor_server import serve

            logger.info(f"Starting sensor server on {args.host}:{args.port}")
            if args.config:
                serve(args.host, args.port, str(args.config))
            else:
                serve(args.host, args.port)

        elif args.command == "simulator":
            from tauro_edge.server.simulator_server import serve

            logger.info(f"Starting simulator server on {args.host}:{args.port}")
            serve(args.host, args.port, args.config)

    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server failed: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
