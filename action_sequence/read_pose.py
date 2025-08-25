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
    main()