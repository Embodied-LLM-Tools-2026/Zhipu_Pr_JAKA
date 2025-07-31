import json
import os

class ObjectLocalization:
    def __init__(self):
        """初始化对象定位类"""
        self.obj_name = None
        self.num = None
        self.layer_mapping = {
            "奶茶": 2,
            "柠檬茶": 3,
            "雪碧": 4,
            "可乐": 5
        }
    
    def get_layer_number(self, json_file_path):
        """
        获取层数方法
        输入：json文件路径，包含obj_name和num两个键
        输出：根据objectname对应的层号
        """
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
            
            # 根据对象名称返回对应的层号
            if self.obj_name in self.layer_mapping:
                layer_number = self.layer_mapping[self.obj_name]
                print(f"对象 '{self.obj_name}' 对应第 {layer_number} 层")
                return layer_number
            else:
                print(f"警告：未找到对象 '{self.obj_name}' 的层号映射")
                return None
                
        except json.JSONDecodeError as e:
            print(f"JSON文件格式错误：{e}")
            return None
        except Exception as e:
            print(f"读取文件时发生错误：{e}")
            return None
    
    def move_to_observable_state(self, layer_number):
        """
        移动到可观测状态方法
        输入：层数
        输出：头部移动角度和身体上下移动距离
        """
        if layer_number is None:
            print("错误：层数不能为空")
            return None, None
        
        # 根据层数计算移动参数
        # 这里使用简单的线性映射，您可以根据实际需求调整
        head_angle = (layer_number - 2) * 15  # 每层增加15度
        body_distance = (layer_number - 2) * 0.1  # 每层增加0.1米
        
        print(f"第 {layer_number} 层的移动参数：")
        print(f"  头部移动角度：{head_angle} 度")
        print(f"  身体上下移动距离：{body_distance} 米")
        
        return head_angle, body_distance
    
    def observe(self):
        """
        观测方法
        """
        pass

def main():
    """测试ObjectLocalization类的功能"""
    
    # 创建类实例
    obj_loc = ObjectLocalization()
    
    # 测试方法1：获取层数
    #print("=== 测试方法1：获取层数 ===")
    json_file_path = "/home/tanyz/Project/Pr_Stage1/objectLocalization/test_data.json"
    
    # 测试奶茶
    layer_number = obj_loc.get_layer_number(json_file_path)
    print(f"获取到的层数：{layer_number}")
    #print()
    
    # 测试方法2：移动到可观测状态
    #print("=== 测试方法2：移动到可观测状态 ===")
    if layer_number is not None:
        head_angle, body_distance = obj_loc.move_to_observable_state(layer_number)
        #print(f"返回的头部角度：{head_angle}")
        #print(f"返回的身体距离：{body_distance}")
    print()
    
    # 测试方法3：观测
    #print("=== 测试方法3：观测 ===")
    obj_loc.observe()
    print("观测方法执行完成（pass）")
    print()
    

if __name__ == "__main__":
    main() 