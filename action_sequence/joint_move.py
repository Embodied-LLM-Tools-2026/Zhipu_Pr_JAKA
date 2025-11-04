# workspace_zone_controller.py
from jaka_msgs.srv import Move

class WorkspaceZoneController:
    def __init__(self, node):
        self.node = node
        self.move_client = self.node.create_client(Move, "/jaka_driver/joint_move")
        while not self.move_client.wait_for_service(timeout_sec=2.0):
            self.node.get_logger().info("等待 JointMove 服务...")

        self.zone_poses = {
            "high": [1.8,-0.763,-1.176,-0.159,-1.0,-0.57],
            "mid": [1.4,-0.763,-1.176,-0.159,-1.0,-0.87],
            "low": [1.3,-0.663,-1.176,-0.159,-1.7,-0.87],
        }

    def move_to_zone(self, zone_name: str):
        zone_name = zone_name.strip().lower()
        if zone_name not in self.zone_poses:
            self.node.get_logger().warn(f"❌ 未识别的工作区: {zone_name}")
            return False

        joint_target = self.zone_poses[zone_name]
        self.node.get_logger().info(f"📍 正在移动到 {zone_name}: {joint_target}")

        request = Move.Request()
        request.pose = joint_target
        request.has_ref = False
        request.ref_joint = [0.0]
        request.mvvelo = 5.0
        request.mvacc = 5.0
        request.mvtime = 0.0
        request.mvradii = 0.0
        request.coord_mode = 0
        request.index = 0

        future = self.move_client.call_async(request)
        # 阻塞等待结果
        import rclpy
        rclpy.spin_until_future_complete(self.node, future)
        try:
            result = future.result()
            if result and result.ret:
                self.node.get_logger().info(f"✅ 成功移动到 {zone_name}: {result.message}")
                return True
            else:
                self.node.get_logger().warn(f"❌ 移动失败: {zone_name} - {result.message if result else '无响应'}")
                return False
        except Exception as e:
            self.node.get_logger().error(f"❌ 移动异常: {zone_name} - {e}")
            return False