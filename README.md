# JAKA 机器人智能语音控制系统 (JAKA Robot Intelligent Voice Control System)

本项目是一个基于 JAKA 机械臂的智能语音控制系统，集成了语音识别 (ASR)、语音合成 (TTS)、视觉语言模型 (VLM) 以及动作执行模块。系统能够理解自然语言指令，通过 VLM 进行场景感知和任务规划，并利用 VLA (Vision-Language-Action) 技术完成复杂的抓取和操作任务。

## ✨ 主要功能 (Key Features)

- **多模态语音交互**: 集成 VAD、SenseVoice ASR 和 Kokoro TTS，实现流畅的语音对话控制。
- **长程任务规划 (Long-Horizon Planning)**: 具备将模糊、复杂的指令（如“把桌子收拾干净”）自动分解为可执行子任务序列的能力。
- **VLA 泛化技能**:
  - `pick_vla`: 基于视觉语言对齐的智能抓取。
  - `place_vla`: 基于视觉语言对齐的智能放置。
  - `vla_execute`: 通用的视觉动作执行（如“擦拭”、“推开”）。
- **智能感知**: 集成 Orbbec 深度相机与 SAM (Segment Anything Model) 进行物体定位与分割。

## 🛠️ 环境准备 (Installation)

### 1. 基础环境
建议使用 Conda 管理环境：

```bash
conda create --name pr python=3.10
conda activate pr
pip install -r requirements.txt
```

### 2. 模型准备
下载必要的语音和 VLM 模型权重：

```bash
# 下载 VAD 模型
python -c "import torch; torch.hub.load('snakers4/silero-vad', 'silero_vad')"

# 下载 TTS 模型 (Kokoro)
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download hexgrad/Kokoro-82M-v1.1-zh --local-dir ./ckpts/kokoro-v1.1
```

### 3. 硬件驱动安装

#### Windows 安装 xapi (JAKA 控制器接口)
```bash
cd whl
pip install xapi-3.1.11-cp310-cp310-linux_x86_64.whl  # 请根据实际系统选择对应的 whl
```

#### 奥比中光 (Orbbec) 摄像头 SDK 配置
参考链接：[PyOrbbecSDK Installation](https://orbbec.github.io/pyorbbecsdk/source/2_installation/build_the_package.html#install-dependencies)

1. **克隆仓库**
   ```bash
   git clone https://github.com/orbbec/pyorbbecsdk
   ```
2. **安装依赖**
   ```bash
   pip3 install -r requirements.txt
   ```
3. **编译与安装** (Windows 示例)
   - 下载 CMake 和 Visual Studio 2022。
   - 生成 build 文件：
     ```bash
     cd pyorbbecsdk
     mkdir build && cd build
     cmake -Dpybind11_DIR="$(pybind11-config --cmakedir)" ..
     ```
   - 使用 VS 打开 `.sln` 文件，编译生成 Release 版本。
   - 生成并安装 Wheel 包：
     ```bash
     cd ..
     pip install wheel
     python setup.py bdist_wheel
     pip install dist\pyorbbecsdk-xxx-cp39-win_amd64.whl
     ```

## 🚀 快速开始 (Quick Start)

### 启动主程序
启动集成了语音、视觉和机械臂控制的主程序：

```bash
python main_hand.py
```

### 启动纯对话测试
仅测试语音交互逻辑（不连接机械臂）：

```bash
python continuous_dialogue.py
```

## 📂 目录结构说明

- `action_sequence/`: 机械臂底层动作脚本 (移动、导航、夹爪控制)。
- `voice/`: 核心智能模块。
  - `agents/`: VLM 代理、规划器 (Planner) 及函数调用处理 (`function_call/`)。
  - `audio/`: ASR, TTS, VAD 等音频处理模块。
  - `control/`: 动作执行器 (`executor.py`) 和技能合约。
  - `perception/`: 视觉感知与定位模块。
- `examples/`: 各类功能的独立测试脚本。
- `tools/`: 相机、日志和 UI 工具。

## 🔧 视觉模块标定 (Calibration)

每当货架上饮料位置或货架与机器人之间相对位置发生变化，需进行标定。

### 1. 修改配置文件
配置文件位于 `config\layer_mapping.json`。
格式：`Key: 饮料名称`, `Value: [层数, 头部俯仰角, 身躯高度]`。

### 2. 运行标定程序
```bash
python objectLocalization/objectDectection/example_usage.py
```
1. 选择 `3. 交互模式`。
2. 选择 `2. 标定新饮料位置`。
3. 按照提示调整相机角度并使用 SAM 进行交互式分割标注。


