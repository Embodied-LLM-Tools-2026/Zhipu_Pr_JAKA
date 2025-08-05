import json
import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from objectLocalization.objectDectection.DrinkShelfLocator import DrinkShelfLocator

class ObjectLocalization:
    def __init__(self):
        """初始化对象定位类"""
        self.obj_name = None
        self.num = None
        self.layer_mapping = {
            "奶茶": [2,28-5],
            "柠檬茶": [3,0,-5],
            "雪碧": [4,0,130],
            "可乐": [5,0,450]
        }

        base_path = os.path.join("objectLocalization", "objectDectection")
        self.locator = DrinkShelfLocator(
            model_path=os.path.join(base_path, "weights/yoloe-11s-seg.pt"),
            reference_dir=os.path.join(base_path, "reference_images"),
            template_dir=os.path.join(base_path, "position_templates"),
            camera_id=0
        )
    
    def get_layer_number(self, json_file_path=None, obj_name=None, num=None):
        """
        获取层数方法
        输入：json文件路径，包含obj_name和num两个键
        输出：根据objectname对应的层号
        """
        if obj_name is None or num is None:
            # 如果obj_name和num为None，则从json文件中读取
            try:
                # 检查文件是否存在
                if not os.path.exists(json_file_path):
                    raise FileNotFoundError(f"文件 {json_file_path} 不存在")
                
                # 读取JSON文件
                with open(json_file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                
                # 检查必需的键是否存在
                if 'obj_name' not in data or 'num' not in data:
                    raise KeyError("JSON文件必须包含 'obj_name' 和 'num' 两个键")
                # 存储值到类属性中
                self.obj_name = data['obj_name']
                self.num = data['num']
            except json.JSONDecodeError as e:
                print(f"JSON文件格式错误：{e}")
                return None
            except Exception as e:
                print(f"读取文件时发生错误：{e}")
                return None
        else:
            self.obj_name = obj_name
            self.num = num
        
        # 根据对象名称返回对应的层号
        if self.obj_name in self.layer_mapping:
            layer_number = self.layer_mapping[self.obj_name][0]
            head_angle = self.layer_mapping[self.obj_name][1]
            body_distance = self.layer_mapping[self.obj_name][2]
            print(f"对象 '{self.obj_name}' 对应第 {layer_number} 层")
            return layer_number, head_angle, body_distance
        else:
            print(f"警告：未找到对象 '{self.obj_name}' 的层号映射")
            return None
    def observe(self, obj_name, quantity):
        """
        观测方法，查找指定类型的饮料。
        该方法利用DrinkShelfLocator来定位饮料。
        :param obj_name: 要查找的饮料类型 (例如: "雪碧")
        :param quantity: 要查找的数量
        :return: 包含推荐位置编号的列表，如果查找失败则返回None
        """
        search_result = self.locator.find_drinks(obj_name, quantity, grabbing_direction="right", total_drinks=5)
        
        if search_result and search_result.get("success"):
            return search_result.get("positions")
        else:
            # message = search_result.get("message", "未知错误")
            # print(f"查找失败: {message}")
            return []

def main():
    """测试ObjectLocalization类的功能"""
    
    # 创建类实例
    obj_loc = ObjectLocalization()
    
    # 测试方法1：获取层数
    #print("=== 测试方法1：获取层数 ===")
    json_file_path = "/home/tanyz/Project/Pr_Stage1/objectLocalization/test_data.json"
    
    # 测试
    # layer_number,head_angle,body_distance = obj_loc.get_layer_number(json_file_path)
    # print(f"{obj_loc.obj_name}所在的层数：{layer_number}, 机器人头部俯仰角：{head_angle}, 机器人身躯高度：{body_distance}")
    #print()

    obj_loc.observe("可乐", 3)
    

if __name__ == "__main__":
    main() 
