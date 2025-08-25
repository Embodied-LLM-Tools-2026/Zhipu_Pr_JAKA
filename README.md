# 人形PR 阶段一

## 环境准备

```bash
conda create --name pr python=3.10
pip install -r requirements.txt
python -c "import torch; torch.hub.load('snakers4/silero-vad', 'silero_vad')" #下载VAD模型
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download hexgrad/Kokoro-82M-v1.1-zh --local-dir ./ckpts/kokoro-v1.1
```
## windows 安装xapi
```bash
cd whl
pip install xapi-3.1.11-cp310-cp310-linux_x86_64.whl
```



## 奥比中光摄像头SDK配置（Windows）
参考链接：https://orbbec.github.io/pyorbbecsdk/source/2_installation/build_the_package.html#install-dependencies


### 1. 克隆仓库
```bash
git clone https://github.com/orbbec/pyorbbecsdk
```
### 2. 创建虚拟环境（注意python版本）或切换到所用环境

### 3. 安装依赖
```bash
pip3 install -r requirements.txt
```
### 4. 下载cmake和Visual Studio 2022

### 5. 生成build文件
```bash
cd pyorbbecsdk
mkdir build && cd build
cmake -Dpybind11_DIR="$(pybind11-config --cmakedir)" ..
```
### 6. 生成lib文件
用vs打开build下的sln文件，左侧解决方案资源管理器，右键“pyorbbecsdk”，点击“重新生成”，右键“install”，点击“生成”。
### 7. 生成whl
```bash
# 进入SDK源码根目录
cd C:\path\to\pyorbbecsdk
# 安装wheel工具
pip install wheel
# 生成Wheel包（依赖编译后的文件）
python setup.py bdist_wheel
```

### 8. 安装
```bash
# 安装Wheel包（替换为实际的文件名）
pip install dist\pyorbbecsdk-xxx-cp39-win_amd64.whl
```

## 视觉模块饮料位置标定
每当货架上饮料位置或货架与机器人之间相对位置发生变化，则需进行该步骤

### 1. 修改饮料位置的配置文件

配置文件位于 `config\layer_mapping.json`

该文件的结构如下：  

Key：饮料名称（字符串类型）

Value：一个包含 3 个整数的列表，分别对应：
[饮料层数, 头部俯仰角, 身躯高度]

其中： 

- 饮料名称（如“水”、“雪碧”等，必须与`voice\config.py`中`drink_list`的元素严格一致）
- 饮料层数（从下往上数）
- 头部俯仰角（正方向为向下旋转）
- 身躯高度（数值越大，身躯高度越高）

### 2. 重新标定饮料位置
如果货架与机器人之间相对位置没变，那么只需要重新标定位置改变了的饮料，否则要重新标定所有饮料

- 首先将所有需要标定的饮料都在货架上摆满，饮料位置要与实际抓取时的一致
- 然后运行`objectLocalization\objectDectection\example_usage.py`
- 在终端中选择运行模式`3. 交互模式`
- 如果机器人不在货架抓取位置处，则先选择`4. 移动机器人到货架处`
- 接着选择`2. 标定新饮料位置`
- 输入要标定的饮料类型（可以是配置文件中定义的任意一个饮料名称，如“水”）
- 等待一段时间，直到终端输出以下信息：
```
    预览控制说明：
    - 上方向键 或 W键：增加俯仰角
    - 下方向键 或 S键：减少俯仰角
    - 回车键：保存图像并继续
    - ESC键：跳过保存直接继续
    - 如果方向键不响应，请尝试使用W/S键
    - 当前俯仰角: 28°
```
- 此时任务栏中会出现一个窗口，它会实时显示当前机器人相机拍摄的画面，请按照预览控制说明调整相机角度，直至所有要标定的饮料都出现在画面当中，随后按下`ENTER`键
- 随后再等待一段时间，直到终端输出以下信息：
```
==================================================
SAM Interactive Annotation Tool
==================================================
Instructions:
- Click anywhere on the image to perform segmentation
- Press 'r' key to reset annotation
- Press 's' key to save results (if output path is specified)
- Press 'q' key to quit program
==================================================
```
- 此时任务栏中又会出现一个窗口，它会显示刚才按下`ENTER`键时机器人相机拍摄的图片。点击图片中要标定的饮料（任意一瓶），被点击的物体会被分割出来，并以可视化形式显示分割结果。可以多次点击图片中的不同物体，直至分割出理想的饮料图像，然后按`q`，即完成了该饮料的标定
- 此时终端会回到交互模式，可以仿照以上步骤继续完成剩余饮料的标定


