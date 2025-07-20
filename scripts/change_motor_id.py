import argparse

import can
from CyberGearDriver import CyberGearMotor, CyberMotorMessage

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--old_id", type=int, required=True)
    parser.add_argument("--new_id", type=int, required=True)
    args = parser.parse_args()

    with can.interface.Bus(interface="socketcan", channel="can0") as bus:

        def send_message(message: CyberMotorMessage):
            bus.send(
                can.Message(
                    arbitration_id=message.arbitration_id,
                    data=message.data,
                    is_extended_id=message.is_extended_id,
                )
            )

        # Create the motor controller
        motor = CyberGearMotor(motor_id=args.old_id, send_message=send_message, verbose=True)

        motor.change_motor_id(args.new_id)
