#!/usr/bin/env python3
"""Test script for MuJoCo scene builder."""

import argparse
from pathlib import Path

import mujoco
from mujoco import viewer

from tauro_edge.utils.scene_builder import SceneBuilder


def main():
    parser = argparse.ArgumentParser(description="Test MuJoCo scene builder")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("tauro_edge/configs/sim_two_robots.yaml"),
        help="Path to robot configuration YAML",
    )
    parser.add_argument(
        "--base-scene",
        type=Path,
        help="Override base scene path (optional)",
    )
    parser.add_argument(
        "--save-xml",
        type=Path,
        help="Save generated XML to file (optional)",
    )

    args = parser.parse_args()

    # Load configuration to get base scene
    import yaml

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Determine base scene path
    if args.base_scene:
        base_scene_path = args.base_scene
    else:
        base_scene_name = config.get("scene", {}).get("base_scene", "empty_scene.xml")
        base_scene_path = Path("tauro_edge/sim_scenes") / base_scene_name

    print("Building scene from:")
    print(f"  Base scene: {base_scene_path}")
    print(f"  Robot config: {args.config}")

    # Build scene
    builder = SceneBuilder(base_scene_path, args.config)
    scene_xml = builder.build_scene()

    print(f"\nGenerated scene XML: {scene_xml}")

    # Optionally save XML
    if args.save_xml:
        import shutil

        shutil.copy(scene_xml, args.save_xml)
        print(f"Saved to: {args.save_xml}")

    # Load and visualize
    print("\nLoading scene in MuJoCo viewer...")
    print("Controls:")
    print("  - Mouse: Rotate camera")
    print("  - Scroll: Zoom")
    print("  - Space: Pause/resume")
    print("  - ESC: Exit")

    model = mujoco.MjModel.from_xml_path(scene_xml)
    data = mujoco.MjData(model)

    # Print robot information
    print(f"\nScene contains {model.njnt} joints:")
    for i in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        print(f"  {i}: {joint_name}")

    print(f"\nScene contains {model.nu} actuators:")
    for i in range(model.nu):
        act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"  {i}: {act_name}")

    # Launch viewer
    with viewer.launch_passive(model, data) as v:
        while v.is_running():
            mujoco.mj_step(model, data)
            v.sync()


if __name__ == "__main__":
    main()
