from typing import Literal

from constants import P_MAX, P_MIN, T_MIN, V_MAX, V_MIN

StateValues = (
    ("position", (P_MIN, P_MAX)),
    ("velocity", (V_MIN, V_MAX)),
    ("torque", (T_MIN, T_MIN)),
    ("temperature", None),
)

state_names = [name for name, _ in StateValues]

StateName = Literal["position", "velocity", "torque", "temperature"]
