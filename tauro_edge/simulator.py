#!/usr/bin/env python
"""Entry point for the MuJoCo robot simulator server."""

import argparse
import logging
from pathlib import Path

from tauro_edge.server.simulator_server import serve


def main():
    """Main entry point for simulator."""
    parser = argparse.ArgumentParser(description="Tauro MuJoCo Robot Simulator")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the simulator server to")
    parser.add_argument(
        "--port",
        type=int,
        default=50053,
        help="Port to bind the simulator server to (default: 50053)",
    )
    parser.add_argument(
        "--config", type=Path, help="Path to configuration file for simulated robots"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting Tauro MuJoCo Robot Simulator")
    logger.info(f"Server address: {args.host}:{args.port}")

    if args.config:
        logger.info(f"Using configuration file: {args.config}")

    try:
        serve(args.host, args.port, args.config)
    except KeyboardInterrupt:
        logger.info("Simulator shutdown requested")
    except Exception as e:
        logger.error(f"Simulator failed: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
