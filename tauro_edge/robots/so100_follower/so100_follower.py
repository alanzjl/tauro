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
from typing import Any

import numpy as np

from tauro_common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from tauro_common.kinematics.mink_kinematics import MinkKinematics
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
            MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
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

        # Initialize kinematics for end effector control
        self.kinematics = MinkKinematics()
        self.ee_frame = "Wrist_Pitch_Roll"  # MuJoCo body name
        self.current_ee_pos = None
        self.current_joint_pos = None
        logger.info("Using Mink-based inverse kinematics")

    @property
    def motor_configs(self) -> dict[str, Any]:
        return self.bus.motors

    @property
    def observation_space(self) -> dict[str, dict]:
        """Return observation space in OpenAI Gym format."""
        motor_names = list(self.bus.motors.keys())

        # Joint space observation
        joint_space = {
            "position": {
                motor: {"shape": (), "dtype": np.dtype(np.float32), "names": None}
                for motor in motor_names
            },
            "velocity": {
                motor: {"shape": (), "dtype": np.dtype(np.float32), "names": None}
                for motor in motor_names
            },
        }

        # End effector observation
        end_effector_space = {
            "position": {
                "shape": (3,),
                "dtype": np.dtype(np.float32),
                "names": ["x", "y", "z"],
            },
            "orientation": {
                "shape": (9,),
                "dtype": np.dtype(np.float32),
                "names": None,
            },
        }

        return {
            "joint": joint_space,
            "end_effector": end_effector_space,
        }

    @property
    def action_space(self) -> dict[str, dict]:
        """Return action space in OpenAI Gym format."""
        motor_names = list(self.bus.motors.keys())
        joint_space = {
            f"{motor}.pos": {"shape": (), "dtype": np.dtype(np.float32), "names": None}
            for motor in motor_names
        }
        # Support both joint and end effector control
        return {
            # Joint control
            **joint_space,
            # End effector control
            "end_effector": {
                "shape": (4,),  # delta_x, delta_y, delta_z, gripper
                "dtype": np.dtype(np.float32),
                "names": ["delta_x", "delta_y", "delta_z", "gripper"],
            },
        }

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

        # Read arm position and velocity
        start = time.perf_counter()
        position_dict = self.bus.sync_read("Present_Position")
        velocity_dict = self.bus.sync_read("Present_Velocity")

        # Build joint observation dictionary
        joint_obs = {
            "position": {motor: position_dict[motor] for motor in self.bus.motors},
            "velocity": {motor: velocity_dict[motor] for motor in self.bus.motors},
        }

        # Calculate end effector state
        motor_positions = [
            position_dict[name] for name in list(self.bus.motors.keys())[:-1]
        ]  # Exclude gripper
        joint_positions = np.array(motor_positions)
        ee_pos = self.kinematics.forward_kinematics(joint_positions, frame=self.ee_frame)

        # Build end effector observation dictionary
        end_effector_obs = {
            "position": ee_pos[:3, 3],  # Extract x, y, z position
            "orientation": ee_pos[:3, :3].flatten(),  # 3x3 rotation matrix as flat array
        }

        # Update current state for end effector control (include gripper)
        all_joint_positions = [position_dict[name] for name in self.bus.motors]
        self.current_joint_pos = np.array(all_joint_positions)
        self.current_ee_pos = ee_pos

        # Build final observation dictionary
        obs_dict = {
            "joints": joint_obs,
            "end_effector": end_effector_obs,
        }

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
        if "end_effector" in action:
            return self._send_end_effector_action(action["end_effector"])

        # Otherwise, process as joint action
        goal_pos = action["joints"]["position"]

        # Cap goal position when too far away from present position.
        # /!\ Slower fps expected due to reading from the follower.
        if self.config.max_relative_target is not None:
            present_pos = self.bus.sync_read("Present_Position")
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        # Send goal position to the arm
        self.bus.sync_write("Goal_Position", goal_pos)

    def _send_end_effector_action(self, ee_action: dict[str, Any] | np.ndarray) -> dict[str, Any]:
        """Send end effector space action."""
        logger.debug(f"Received end effector action: {ee_action}")
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
            logger.debug(f"Read joint positions: {current_joint_pos}")
            logger.debug(f"Motor names: {list(self.bus.motors.keys())}")
            self.current_joint_pos = np.array(
                [current_joint_pos[name] for name in self.bus.motors], dtype=np.float32
            )
            logger.debug(f"Current joint pos array shape: {self.current_joint_pos.shape}")

        if self.current_ee_pos is None:
            self.current_ee_pos = self.kinematics.forward_kinematics(
                self.current_joint_pos[:-1],
                frame=self.ee_frame,  # Exclude gripper
            )

        logger.debug(f"Current EE position: {self.current_ee_pos[:3, 3]}")
        logger.debug(f"Delta EE (after scaling): {delta_ee}")

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

        # Compute inverse kinematics using mink
        logger.debug(
            f"Current joint pos (excluding gripper): {self.current_joint_pos[:-1] if self.current_joint_pos is not None else None}"
        )
        logger.debug(f"Target EE position: {desired_ee_pos[:3, 3]}")

        target_joint_values = self.kinematics.solve_ik(
            target_position=desired_ee_pos[:3, 3],
            current_joint_pos_deg=self.current_joint_pos[:-1]
            if self.current_joint_pos is not None
            else None,
            frame=self.ee_frame,
            position_weight=1.0,
            orientation_weight=0.0,  # Position only for now
        )

        logger.debug(f"IK solution: {target_joint_values}")

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

        # Don't update current_ee_pos yet - let get_observation do it
        # Only update joint positions for tracking
        if self.current_joint_pos is None:
            self.current_joint_pos = np.zeros(len(self.bus.motors))
        self.current_joint_pos[:-1] = target_joint_values
        self.current_joint_pos[-1] = joint_action["gripper.pos"]

        # Send joint action
        return self.send_action(joint_action)

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.disconnect(self.config.disable_torque_on_disconnect)

        logger.info(f"{self} disconnected.")
