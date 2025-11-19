#!/usr/bin/env python3
"""
Simple helper to drive the Jaka arm to an absolute TCP pose by
1) querying /jaka_driver/get_ik for a joint solution (with ref joints)
2) calling /jaka_driver/joint_move to execute it.

Example:
    python examples/move_arm_via_ik.py --pose 420,0,380,3.14,0,0
    python examples/move_arm_via_ik.py --pose 300,120,400,0,1.57,0 --ref-joints -0.5 -0.8 -0.9 0 -1.0 0.4
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Sequence

import rclpy
from rclpy.node import Node
from rclpy.task import Future

from jaka_msgs.srv import GetIK, Move
from sensor_msgs.msg import JointState


def _parse_pose(values: Sequence[float] | str) -> List[float]:
    if isinstance(values, str):
        parts = [part.strip() for part in values.split(",") if part.strip()]
        if len(parts) != 6:
            raise ValueError("Pose string must be 'x,y,z,rx,ry,rz'")
        return [float(part) for part in parts]
    if len(values) != 6:
        raise ValueError("Pose must contain exactly 6 numbers: x y z rx ry rz")
    return [float(v) for v in values]


class IKMoveNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("ik_move_helper")
        self.args = args
        self.joint_move_client = self.create_client(Move, "/jaka_driver/joint_move")
        self.get_ik_client = self.create_client(GetIK, "/jaka_driver/get_ik")
        self.joint_state_future: Future[JointState] | None = None
        if args.ref_joints is None:
            self.joint_state_future = Future()
            self.create_subscription(
                JointState,
                "/jaka_driver/joint_position",
                self._joint_state_callback,
                qos_profile=10,
            )

    def _joint_state_callback(self, msg: JointState) -> None:
        if self.joint_state_future is None or self.joint_state_future.done():
            return
        if len(msg.position) >= 6:
            self.joint_state_future.set_result(msg)

    def wait_for_services(self) -> None:
        clients = [
            (self.joint_move_client, "/jaka_driver/joint_move"),
            (self.get_ik_client, "/jaka_driver/get_ik"),
        ]
        for client, name in clients:
            if not client.wait_for_service(timeout_sec=5.0):
                raise RuntimeError(f"Service {name} not available")

    def resolve_ref_joints(self) -> List[float]:
        if self.args.ref_joints is not None:
            return [float(v) for v in self.args.ref_joints]
        if self.joint_state_future is None:
            raise RuntimeError("Internal error: missing joint_state_future")
        rclpy.spin_until_future_complete(self, self.joint_state_future, timeout_sec=self.args.joint_state_timeout)
        if not self.joint_state_future.done():
            raise RuntimeError("Timeout waiting for /jaka_driver/joint_position")
        msg = self.joint_state_future.result()
        if msg is None or len(msg.position) < 6:
            raise RuntimeError("Received invalid JointState message")
        return list(msg.position[:6])

    def call_get_ik(self, ref_joints: Sequence[float]) -> List[float]:
        request = GetIK.Request()
        request.cartesian_pose = self.args.pose
        request.ref_joint = list(ref_joints)
        future = self.get_ik_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=self.args.service_timeout)
        if not future.done():
            raise RuntimeError("get_ik request timed out")
        response = future.result()
        if response is None:
            raise RuntimeError("get_ik returned no response")
        if not response.joint or any(abs(val) >= 9999 for val in response.joint):
            raise RuntimeError(f"get_ik failed: {response.message}")
        return list(response.joint[:6])

    def call_joint_move(self, joint_target: Sequence[float]) -> None:
        request = Move.Request()
        request.pose = list(joint_target)
        request.has_ref = False
        request.ref_joint = [0.0]
        request.mvvelo = self.args.speed
        request.mvacc = self.args.acc
        request.mvtime = 0.0
        request.mvradii = 0.0
        request.coord_mode = 0
        request.index = 0
        future = self.joint_move_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=self.args.service_timeout)
        if not future.done():
            raise RuntimeError("joint_move request timed out")
        response = future.result()
        if response is None or not response.ret:
            msg = response.message if response else "no response"
            raise RuntimeError(f"joint_move failed: {msg}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Drive Jaka arm to a TCP pose via get_ik + joint_move")
    parser.add_argument(
        "--pose",
        type=str,
        required=True,
        help="Target TCP pose formatted as 'x,y,z,rx,ry,rz' (mm/rad, RPY).",
    )
    parser.add_argument(
        "--ref-joints",
        nargs=6,
        type=float,
        help="Optional joint reference (rad). If omitted the script samples /jaka_driver/joint_position once.",
    )
    parser.add_argument("--speed", type=float, default=5.0, help="Joint move velocity (rad/s).")
    parser.add_argument("--acc", type=float, default=5.0, help="Joint move acceleration (rad/s^2).")
    parser.add_argument("--joint-state-timeout", type=float, default=3.0, help="Seconds to wait for joint_state.")
    parser.add_argument("--service-timeout", type=float, default=10.0, help="Seconds to wait for service responses.")
    args = parser.parse_args(argv)
    args.pose = _parse_pose(args.pose)

    rclpy.init(args=None)
    node = IKMoveNode(args)
    try:
        node.wait_for_services()
        ref_joints = node.resolve_ref_joints()
        node.get_logger().info(f"Using reference joints: {[round(v, 4) for v in ref_joints]}")
        joint_target = node.call_get_ik(ref_joints)
        node.get_logger().info(f"get_ik returned: {[round(v, 4) for v in joint_target]}")
        node.call_joint_move(joint_target)
        node.get_logger().info("Joint move command accepted.")
    except Exception as exc:
        node.get_logger().error(str(exc))
        rclpy.shutdown()
        return 1
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
