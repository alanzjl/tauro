#!/usr/bin/env python3
"""Compile protobuf files for the Tauro project."""

import subprocess
import sys
from pathlib import Path


def compile_proto():
    """Compile all .proto files in tauro_common/proto directory."""
    project_root = Path(__file__).parent.parent
    proto_dir = project_root / "tauro_common" / "proto"

    # Ensure proto directory exists
    if not proto_dir.exists():
        print(f"Error: Proto directory {proto_dir} does not exist")
        return False

    # Find all .proto files
    proto_files = list(proto_dir.glob("*.proto"))
    if not proto_files:
        print(f"No .proto files found in {proto_dir}")
        return False

    print(f"Found {len(proto_files)} proto file(s) to compile")

    # Compile each proto file
    for proto_file in proto_files:
        print(f"Compiling {proto_file.name}...")

        # Compile for Python with proper import paths
        cmd = [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            f"--proto_path={project_root}",
            f"--python_out={project_root}",
            f"--grpc_python_out={project_root}",
            f"tauro_common/proto/{proto_file.name}",
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Error compiling {proto_file.name}:")
                print(result.stderr)
                return False
            print(f"  ✓ Successfully compiled {proto_file.name}")
        except Exception as e:
            print(f"Error running protoc: {e}")
            return False

    print("\nAll proto files compiled successfully!")
    return True


if __name__ == "__main__":
    success = compile_proto()
    sys.exit(0 if success else 1)
