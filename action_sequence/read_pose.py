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

def main():
    handle_l = x5.connect("192.168.1.9")
    handle_r = x5.connect("192.168.1.10")

    cjoint = str(x5.get_cjoint(handle_l))
    cpos = str(x5.get_cpoint(handle_l))
    print(format_joints(cjoint))
    print(cpos)
    print("+++++++++++++++++++++++++++++++++++++++++")
    cjoint = str(x5.get_cjoint(handle_r))
    cpos = str(x5.get_cpoint(handle_r))
    print(format_joints(cjoint))
    print(cpos)
if __name__ == "__main__":
    main()