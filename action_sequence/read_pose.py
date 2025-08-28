import xapi.api as x5

def format_joints(input_str):
    # 分割输入字符串为键值对列表
    pairs = input_str.split('; ')
    
    # 处理每个键值对
    formatted_pairs = []
    for pair in pairs:
        key, value = pair.split(': ')
        # 格式化数值：保留3位小数，去掉多余的0和小数点
        formatted_value = f"{float(value):.3f}".rstrip('0').rstrip('.') if '.' in f"{float(value):.3f}" else f"{float(value):.0f}"
        # 特殊处理0值
        if formatted_value == '-0':
            formatted_value = '0'
        elif formatted_value == '0.000':
            formatted_value = '0'
        # 组合成新的键值对
        formatted_pairs.append(f"{key} = {formatted_value}")
    
    # 用逗号连接所有键值对
    return ', '.join(formatted_pairs)

def parse_pose_data(input_str):
    # 定位起始位置：第一个左括号后的内容
    start = input_str.find('(')
    if start == -1:
        raise ValueError("No '(' found in input string")
    
    # 定位结束位置：找到对应的右括号
    depth = 1
    end = start + 1
    while end < len(input_str) and depth > 0:
        if input_str[end] == '(':
            depth += 1
        elif input_str[end] == ')':
            depth -= 1
        end += 1
    
    if depth > 0:
        raise ValueError("Mismatched parentheses in input string")
    
    # 提取括号内的内容
    content = input_str[start + 1:end - 1]
    
    # 分割键值对
    items = []
    for pair in content.split(','):
        pair = pair.strip()  # 去除前后空格
        if not pair:
            continue
        
        # 分离键和值
        if ':' not in pair:
            raise ValueError(f"Invalid key-value pair: {pair}")
        
        key, value = pair.split(':', 1)  # 只分割第一个冒号
        key = key.strip()
        value = value.strip()
        
        items.append((key, value))
    
    # 构建格式化字符串
    formatted = []
    required_keys = ['x', 'y', 'z', 'a', 'b', 'c', 'e1', 'e2', 'e3']
    
    for key in required_keys:
        for k, v in items:
            if k == key:
                formatted.append(f"{key}={v}")
                break
        else:
            raise ValueError(f"Missing required key: {key}")
    
    return ', '.join(formatted)

def add_pick_point(point_index, point_meaning, side='L', ip_left="192.168.1.9", ip_right="192.168.1.10"):
    """
    读取当前机器人关节数据，格式化为 PP_hand.py 风格的 Joint 定义字符串并打印。

    :param point_index: 第几个点位（如 2, 3, 4...）
    :param point_meaning: 该点位的意义（字符串）
    :param side: 选择读取哪只手的关节数据，'L' 或 'R'
    :param ip_left: 左手机器人 IP
    :param ip_right: 右手机器人 IP
    """
    # 连接机器人
    handle_l = x5.connect(ip_left)
    handle_r = x5.connect(ip_right)

    # 读取当前关节（与 main 中方式一致）
    if str(side).upper() == 'R':
        cjoint = str(x5.get_cjoint(handle_r))
    else:
        cjoint = str(x5.get_cjoint(handle_l))

    # 使用已有的 format_joints 进行格式化（得到 "j1 = ..., j2 = ..." 的一行）
    formatted = format_joints(cjoint)

    # 拆分参数，按 PP_hand.py 风格换行对齐（前4个在第一行，其余在第二行，22个空格缩进）
    args = [seg.strip() for seg in formatted.split(',')]
    first_line_args = ', '.join(args[:4])
    second_line_args = ', '.join(args[4:])

    # 构造两行字符串
    line1 = f"pick_{point_index} = x5.Joint({first_line_args}, "
    line2 = f"{' ' * 22}{second_line_args})"

    # 打印注释与结果
    print(f"# {point_meaning}")
    print(line1)
    print(line2)
    str_movJ = f"x5.movj(handle_{side}, pick_{point_index}, add_data)"
    str_wait_move_done = f"x5.wait_move_done(handle_{side})"
    print(str_movJ)
    print(str_wait_move_done)

def main():
    handle_l = x5.connect("192.168.1.9")
    handle_r = x5.connect("192.168.1.10")

    cjoint = str(x5.get_cjoint(handle_l))
    cpos = str(x5.get_cpoint(handle_l))
    print(format_joints(cjoint))
    print(parse_pose_data(cpos))
    print("+++++++++++++++++++++++++++++++++++++++++")
    cjoint = str(x5.get_cjoint(handle_r))
    cpos = str(x5.get_cpoint(handle_r))
    print(format_joints(cjoint))
    print(parse_pose_data(cpos))
if __name__ == "__main__":
    # 示例：读取左手当前关节并输出 PP_hand.py 风格 Joint 定义
    # add_pick_point(4, "收回点位", side='R')
    # add_pick_point(2, "预抓取点位2", side='R')
    # add_pick_point(3, "抓取点位", side='R')

    # add_pick_point(2, "预抓取点位", side='R')
    # add_pick_point(3, "抓取点位", side='R')
    # add_pick_point(4, "回收点位", side='R')

    # add_pick_point(2, "预抓取点位", side='L')
    # add_pick_point(3, "抓取点位", side='L')
    add_pick_point(2, "收回点位1", side='L')