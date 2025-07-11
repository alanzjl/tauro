#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
from functools import cached_property
from typing import Any

import numpy as np

from tauro_common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from tauro_common.model.kinematics import RobotKinematics
from tauro_edge.motors import Motor, MotorCalibration, MotorNormMode
from tauro_edge.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_so100_follower import SO100FollowerConfig

logger = logging.getLogger(__name__)


class SO100Follower(Robot):
    """
    [SO-100 Follower Arm](https://github.com/TheRobotStudio/SO-ARM100) designed by TheRobotStudio
    """

    config_class = SO100FollowerConfig
    name = "so100_follower"

    def __init__(self, config: SO100FollowerConfig):
        super().__init__(config)
        self.config = config
        norm_mode_body = (
            MotorNormMode.DEGREES
            if config.use_degrees or config.enable_end_effector_control
            else MotorNormMode.RANGE_M100_100
        )
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "wrist_roll": Motor(5, "sts3215", norm_mode_body),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )

        # Initialize kinematics if end effector control is enabled
        if config.enable_end_effector_control:
            self.kinematics = RobotKinematics(robot_type="so_new_calibration")
            self.current_ee_pos = None
            self.current_joint_pos = None
            self.ee_frame = "gripper_tip"

    @property
    def motor_configs(self) -> dict[str, Any]:
        return self.bus.motors

    @property
    def observation_space(self) -> dict[str, dict]:
        """Return observation space in OpenAI Gym format."""
        motor_names = list(self.bus.motors.keys())
        return {
            f"{motor}.pos": {"shape": (), "dtype": np.dtype(np.float32), "names": None}
            for motor in motor_names
        }

    @property
    def action_space(self) -> dict[str, dict]:
        """Return action space in OpenAI Gym format."""
        if self.config.enable_end_effector_control:
            # End effector control space
            return {
                "end_effector": {
                    "shape": (4,),  # delta_x, delta_y, delta_z, gripper
                    "dtype": np.dtype(np.float32),
                    "names": ["delta_x", "delta_y", "delta_z", "gripper"],
                },
                "joints": self.observation_space,  # Also support joint control
            }
        else:
            # Joint control space only
            return self.observation_space

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return self._motors_ft

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    def connect(self, calibrate: bool = True) -> None:
        """
        We assume that at connection time, arm is in a rest position,
        and torque can be safely disabled to run calibration.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus.connect()
        if not self.is_calibrated and calibrate:
            self.calibrate()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        logger.info(f"\nRunning calibration of {self}")
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input(f"Move {self} to the middle of its range of motion and press ENTER....")
        homing_offsets = self.bus.set_half_turn_homings()

        full_turn_motor = "wrist_roll"
        unknown_range_motors = [motor for motor in self.bus.motors if motor != full_turn_motor]
        print(
            f"Move all joints except '{full_turn_motor}' sequentially through their "
            "entire ranges of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion(unknown_range_motors)
        range_mins[full_turn_motor] = 0
        range_maxes[full_turn_motor] = 4095

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)

    def configure(self) -> None:
        with self.bus.torque_disabled():
            self.bus.configure_motors()
            for motor in self.bus.motors:
                self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
                # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
                self.bus.write("P_Coefficient", motor, 16)
                # Set I_Coefficient and D_Coefficient to default value 0 and 32
                self.bus.write("I_Coefficient", motor, 0)
                self.bus.write("D_Coefficient", motor, 32)

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read arm position
        start = time.perf_counter()
        obs_dict = self.bus.sync_read("Present_Position")
        obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}

        # Add end effector position if enabled
        if self.config.enable_end_effector_control:
            joint_positions = np.array([obs_dict[f"{name}.pos"] for name in self.bus.motors])
            ee_pos = self.kinematics.forward_kinematics(joint_positions, frame=self.ee_frame)
            obs_dict["end_effector.pos"] = ee_pos[:3, 3]  # Extract x, y, z position
            obs_dict["end_effector.orientation"] = ee_pos[:3, :3].flatten()  # 3x3 rotation matrix

            # Update current state for end effector control
            self.current_joint_pos = joint_positions
            self.current_ee_pos = ee_pos

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")
        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Command arm to move to a target configuration.

        Supports both joint-space and end-effector-space control:
        - Joint control: action contains motor.pos keys
        - End effector control: action contains 'end_effector' key with delta movements

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Raises:
            RobotDeviceNotConnectedError: if robot is not connected.

        Returns:
            the action sent to the motors, potentially clipped.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Check if this is an end effector action
        if "end_effector" in action and self.config.enable_end_effector_control:
            return self._send_end_effector_action(action["end_effector"])

        # Otherwise, process as joint action
        goal_pos = {
            key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")
        }

        # Cap goal position when too far away from present position.
        # /!\ Slower fps expected due to reading from the follower.
        if self.config.max_relative_target is not None:
            present_pos = self.bus.sync_read("Present_Position")
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        # Send goal position to the arm
        self.bus.sync_write("Goal_Position", goal_pos)
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    def _send_end_effector_action(self, ee_action: dict[str, Any] | np.ndarray) -> dict[str, Any]:
        """Send end effector space action."""
        # Convert action to numpy array if dict
        if isinstance(ee_action, dict):
            delta_ee = np.array(
                [
                    ee_action.get("delta_x", 0.0) * self.config.end_effector_step_sizes["x"],
                    ee_action.get("delta_y", 0.0) * self.config.end_effector_step_sizes["y"],
                    ee_action.get("delta_z", 0.0) * self.config.end_effector_step_sizes["z"],
                ],
                dtype=np.float32,
            )
            gripper_action = ee_action.get("gripper", 1.0)
        else:
            # Assume numpy array [delta_x, delta_y, delta_z, gripper]
            delta_ee = ee_action[:3] * np.array(
                [
                    self.config.end_effector_step_sizes["x"],
                    self.config.end_effector_step_sizes["y"],
                    self.config.end_effector_step_sizes["z"],
                ]
            )
            gripper_action = ee_action[3] if len(ee_action) > 3 else 1.0

        # Initialize current state if needed
        if self.current_joint_pos is None:
            current_joint_pos = self.bus.sync_read("Present_Position")
            self.current_joint_pos = np.array([current_joint_pos[name] for name in self.bus.motors])

        if self.current_ee_pos is None:
            self.current_ee_pos = self.kinematics.forward_kinematics(
                self.current_joint_pos, frame=self.ee_frame
            )

        # Calculate desired end effector position
        desired_ee_pos = np.eye(4)
        desired_ee_pos[:3, :3] = self.current_ee_pos[:3, :3]  # Keep orientation
        desired_ee_pos[:3, 3] = self.current_ee_pos[:3, 3] + delta_ee

        # Clip to bounds
        if self.config.end_effector_bounds is not None:
            desired_ee_pos[:3, 3] = np.clip(
                desired_ee_pos[:3, 3],
                self.config.end_effector_bounds["min"],
                self.config.end_effector_bounds["max"],
            )

        # Compute inverse kinematics
        target_joint_values = self.kinematics.ik(
            self.current_joint_pos, desired_ee_pos, position_only=True, frame=self.ee_frame
        )

        target_joint_values = np.clip(target_joint_values, -180.0, 180.0)

        # Create joint action
        joint_action = {
            f"{key}.pos": target_joint_values[i]
            for i, key in enumerate(list(self.bus.motors.keys())[:-1])  # Exclude gripper
        }

        # Handle gripper
        joint_action["gripper.pos"] = np.clip(
            self.current_joint_pos[-1] + (gripper_action - 1) * self.config.max_gripper_pos,
            5,
            self.config.max_gripper_pos,
        )

        # Update current state
        self.current_ee_pos = desired_ee_pos.copy()
        self.current_joint_pos = target_joint_values.copy()
        self.current_joint_pos[-1] = joint_action["gripper.pos"]

        # Send joint action
        return self.send_action(joint_action)

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.disconnect(self.config.disable_torque_on_disconnect)

        logger.info(f"{self} disconnected.")
