# 人形PR 阶段一

## 环境准备

```bash
conda create --name pr python=3.10
pip install -r requirements.txt
python -c "import torch; torch.hub.load('snakers4/silero-vad', 'silero_vad')" #下载VAD模型
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