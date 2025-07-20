#!/usr/bin/env python3
"""Download SO-ARM100 model from MuJoCo Menagerie."""

import sys
import urllib.request
from pathlib import Path


def download_model():
    """Download the SO-ARM100 MuJoCo XML and mesh files."""
    # Determine the target directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    models_dir = project_root / "tauro_common" / "models" / "so_arm100"

    # Create directories
    models_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = models_dir / "assets"
    assets_dir.mkdir(exist_ok=True)

    base_url = (
        "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main/trs_so_arm100"
    )

    # Download MuJoCo XML
    print(f"Downloading MuJoCo XML to {models_dir}...")
    xml_path = models_dir / "so_arm100.xml"
    urllib.request.urlretrieve(f"{base_url}/so_arm100.xml", str(xml_path))
    print(f"✓ Downloaded XML to {xml_path}")

    # Download scene XML
    scene_path = models_dir / "scene.xml"
    urllib.request.urlretrieve(f"{base_url}/scene.xml", str(scene_path))
    print(f"✓ Downloaded scene XML to {scene_path}")

    # Download mesh files
    meshes = [
        "Base.stl",
        "Base_Motor.stl",
        "Rotation_Pitch.stl",
        "Rotation_Pitch_Motor.stl",
        "Upper_Arm.stl",
        "Upper_Arm_Motor.stl",
        "Lower_Arm.stl",
        "Lower_Arm_Motor.stl",
        "Wrist_Pitch_Roll.stl",
        "Wrist_Pitch_Roll_Motor.stl",
        "Fixed_Jaw.stl",
        "Fixed_Jaw_Collision_1.stl",
        "Fixed_Jaw_Collision_2.stl",
        "Fixed_Jaw_Motor.stl",
        "Moving_Jaw.stl",
        "Moving_Jaw_Collision_1.stl",
        "Moving_Jaw_Collision_2.stl",
        "Moving_Jaw_Collision_3.stl",
    ]

    print(f"\nDownloading mesh files to {assets_dir}...")
    for mesh in meshes:
        mesh_path = assets_dir / mesh
        print(f"  Downloading {mesh}...")
        urllib.request.urlretrieve(f"{base_url}/assets/{mesh}", str(mesh_path))

    print("\n✓ Download complete!")
    print(f"MuJoCo model and meshes are located at: {models_dir}")


if __name__ == "__main__":
    try:
        download_model()
    except Exception as e:
        print(f"\n✗ Error downloading model: {e}")
        sys.exit(1)
