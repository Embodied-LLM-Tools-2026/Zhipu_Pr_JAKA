
import csv
import time
import rclpy
from rclpy.node import Node
from jaka_msgs.srv import ServoMove
from jaka_msgs.srv import ServoMove, ServoMoveEnable

class JakaServoClientAsync(Node):
    def __init__(self):
        super().__init__('jaka_servo_p_clientpy')
        self.servo_move_enable_client  = self.create_client(ServoMoveEnable, '/jaka_driver/servo_move_enable')
        self.servo_p_client = self.create_client(ServoMove, '/jaka_driver/servo_p')

        while not self.servo_move_enable_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = ServoMove.Request()

    def enable_servo_mode(self):
        enable_request = ServoMoveEnable.Request()
        enable_request.enable = True
        return self.servo_move_enable_client.call_async(enable_request)

    def send_request(self,pose_delta):# pose_delta[]
        self.req.pose = pose_delta
        return self.servo_p_client.call_async(self.req)
    
class TaskExecutor:
    def __init__(self, navigator, gripper_controller, node: Node):
        self.navigator = navigator  # 导航控制器
        self.gripper = gripper_controller  # 夹爪控制器（由你后续实现）
        self.node = node  # ROS2 Node
        self.servo_client = JakaServoClientAsync()
        future = self.servo_client.enable_servo_mode()
        rclpy.spin_until_future_complete(self.servo_client, future)
        response = future.result()
        self.servo_client.get_logger().info('servo mode enable %s' % (response.message))


    def execute_task(self, task_name):
        """
        执行完整任务流程：导航到目标，机械臂抓取，导航回吧台，递交任务
        task_name: 任务名称，如 '水'、'雪碧'、'可乐'
        """
        # 1. 导航到目标点
        location_map = {
            '水': 'bhc2',
            '可乐': 'col_west_2',
        }
        target_location = location_map.get(task_name, None)
        if not target_location:
            print(f"未知任务名称: {task_name}")
            return False
        print(f"导航到 {target_location} ...")
        self.navigator.navigate_to_target(target_location)
        print(f"导航完成,到达目标点 {target_location}.")

        # 2. 读取对应csv并执行机械臂抓取
        csv_path = f"/home/sht/DIJA/Pr/action_sequence/action_csv/servo_record_bhc_20251029_140852.csv"  # 约定每种任务有对应csv
        print(f"执行抓取动作，读取 {csv_path}")
        self.execute_from_csv(csv_path)

        # 3. 导航回吧台
        print("返回吧台 ...")
        self.navigator.navigate_to_target("bar1")


        # 4. 递交任务
        print(f"递交 {task_name} ...")
        csv_path = f"/home/sht/DIJA/Pr/action_sequence/action_csv/servo_record_bhc_20251029_133013.csv"  # 约定每种任务有对应csv
        print(f"执行抓取动作，读取 {csv_path}")
        self.execute_from_csv(csv_path)

        return True
   

    def execute_from_csv(self, csv_path):
        with open(csv_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            cont=0
            for row in reader:
                cont+=1
                if cont==1: 
                    continue
                if len(row) < 7:
                    print(f"跳过无效行: {row}")
                    continue
                pose = [float(row[i]) for i in range(6)]
                gripper_cmd = float(row[6])
                print(f"执行机械臂动作: pose={pose}, gripper={gripper_cmd}")
                self._servo_p_move(pose)
                if gripper_cmd > 0.5:
                    self.gripper.open()
                else:
                    self.gripper.close()

    def _servo_p_move(self, pose):
        req = ServoMove.Request()
        req.pose = pose
        # req.mvtime = 0.0
        future = self.servo_client.send_request(pose)
    # _wait_navigation 已废弃，导航等待已集成到 navigator

# 用法示例（伪代码，需在 ROS2 节点环境下运行）
# import rclpy
# rclpy.init()
# node = rclpy.create_node('task_executor_node')
# navigator = ... # 你的导航控制器
# gripper_controller = ... # 你的夹爪控制器
# executor = TaskExecutor(navigator, gripper_controller, bar_location="bar", node=node)
# executor.execute_from_csv('tasks.csv')
# rclpy.shutdown()


