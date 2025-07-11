#!/usr/bin/env python3
"""Main entry point for tauro_edge server."""

import argparse
import logging

from tauro_common.constants import DEFAULT_GRPC_HOST, DEFAULT_GRPC_PORT
from tauro_common.utils.utils import init_logging
from tauro_edge.server.robot_server import serve


def main():
    parser = argparse.ArgumentParser(description="Tauro Edge Server - Robot Control Service")
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_GRPC_HOST,
        help=f"Host to bind the server to (default: {DEFAULT_GRPC_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_GRPC_PORT,
        help=f"Port to bind the server to (default: {DEFAULT_GRPC_PORT})",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Initialize logging
    log_level = getattr(logging, args.log_level)
    init_logging()
    logging.getLogger().setLevel(log_level)

    # Start server
    try:
        serve(host=args.host, port=args.port)
    except KeyboardInterrupt:
        logging.info("Server stopped by user")
    except Exception as e:
        logging.error(f"Server error: {e}")
        raise


if __name__ == "__main__":
    main()
