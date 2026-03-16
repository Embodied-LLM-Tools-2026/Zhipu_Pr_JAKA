# SSH 远程连接教程 - 机器人项目专用指南

> **文档版本**：v1.0  
> **创建日期**：2026-03-16  
> **适用对象**：需要远程连接机器人 Linux 系统的开发者  
> **前置知识**：基础的计算机网络知识

---

## 一、SSH 基础概念

### 1.1 什么是 SSH？

**SSH（Secure Shell）** 是一种网络协议，用于**安全地远程登录**到另一台电脑并执行命令。

**你的使用场景**：
```
┌─────────────────┐         SSH 连接          ┌─────────────────┐
│  你的电脑        │ ◄──────────────────────► │  机器人的电脑    │
│  Windows        │         网络通信           │  Linux          │
│                 │                           │                 │
│  - 开发代码      │                           │  - 运行程序      │
│  - 编辑文件      │                           │  - 控制机器人    │
│  - 上传代码      │                           │  - 连接硬件      │
└─────────────────┘                           └─────────────────┘
```

**为什么需要 SSH**：
- ✅ 机器人的 Linux 电脑可能没有显示器、键盘、鼠标
- ✅ 你需要从自己的电脑远程控制它
- ✅ SSH 提供了安全的命令行访问
- ✅ 可以传输文件、运行程序、调试代码

### 1.2 SSH 工作原理

```
你的电脑（Windows）                    机器人电脑（Linux）
      │                                      │
      │  1. 发起 SSH 连接请求                    │
      │─────────────────────────────────────>│
      │                                      │
      │  2. 返回公钥指纹                       │
      │<─────────────────────────────────────│
      │                                      │
      │  3. 确认指纹（首次连接）                 │
      │─────────────────────────────────────>│
      │                                      │
      │  4. 输入密码                           │
      │─────────────────────────────────────>│
      │                                      │
      │  5. 验证通过，建立加密通道               │
      │<═════════════════════════════════════│
      │                                      │
      │  6. 发送命令/传输文件                   │
      │<═════════════════════════════════════>│
```

**关键特点**：
- 🔒 **加密通信**：所有数据传输都经过加密
- 🔑 **身份验证**：支持密码、密钥等多种验证方式
- 🛡️ **防止中间人攻击**：通过公钥指纹验证服务器身份

---

## 二、Windows 上配置 SSH 客户端

### 2.1 检查是否已安装 SSH

打开 **PowerShell**（按 Win+X，选择"Windows PowerShell"），输入：

```powershell
ssh -V
```

**如果显示版本号**（如 `OpenSSH_for_Windows_8.1p1`），说明已安装，可以跳过此节。

**如果提示"ssh : 无法将'ssh'项识别为 cmdlet"**，说明未安装，需要安装。

### 2.2 安装 SSH 客户端

#### 方法 1：通过 Windows 设置（推荐）

1. 打开 **设置** → **应用** → **可选功能**
2. 点击 **"添加功能"**
3. 搜索 **"OpenSSH 客户端"**
4. 点击 **"安装"**
5. 等待安装完成

#### 方法 2：通过 PowerShell（管理员权限）

1. 右键点击开始菜单，选择 **"Windows PowerShell（管理员）"**
2. 输入以下命令：

```powershell
Add-WindowsCapability -Online -Name OpenSSH.Client~~~~0.0.1.0
```

3. 等待安装完成（可能需要几分钟）

#### 验证安装

安装完成后，重新打开 PowerShell，输入：

```powershell
ssh -V
```

应该显示版本信息。

---

## 三、首次 SSH 连接

### 3.1 获取机器人 Linux 电脑的信息

在连接之前，你需要知道：

1. **IP 地址**：机器人 Linux 电脑的网络地址
2. **用户名**：登录 Linux 的用户名
3. **密码**：对应用户的密码

**示例信息**（实际使用时替换为你的）：
- IP 地址：`192.168.1.100`
- 用户名：`robot`
- 密码：`robot123`

### 3.2 确保网络连通

在连接之前，先测试网络是否通畅：

```powershell
ping 192.168.1.100
```

**正常输出**：
```
正在 Ping 192.168.1.100 具有 32 字节的数据:
来自 192.168.1.100 的回复：字节=32 时间=1ms TTL=64
来自 192.168.1.100 的回复：字节=32 时间=1ms TTL=64
来自 192.168.1.100 的回复：字节=32 时间=1ms TTL=64
来自 192.168.1.100 的回复：字节=32 时间=1ms TTL=64

192.168.1.100 的 Ping 统计信息:
    数据包：已发送 = 4，已接收 = 4，丢失 = 0 (0% 丢失)
```

**如果 ping 不通**：
- ❌ 检查你的电脑和机器人是否在**同一局域网**
- ❌ 检查机器人是否**已开机**
- ❌ 检查**防火墙设置**

### 3.3 首次连接步骤

在 PowerShell 中输入：

```powershell
ssh robot@192.168.1.100
```

**连接过程详解**：

```
第 1 步：输入命令
PS C:\Users\YourName> ssh robot@192.168.1.100

第 2 步：首次连接会提示确认指纹
The authenticity of host '192.168.1.100' can't be established.
ECDSA key fingerprint is SHA256:xxxxxxxxxxxxxxxxxxxxxxxxxxx.
This key is not known by any other names
Are you sure you want to continue connecting (yes/no/[fingerprint])? yes

第 3 步：输入 yes 确认
Please type 'yes' or 'no': yes
Warning: Permanently added '192.168.1.100' (ECDSA) to the list of known hosts.

第 4 步：输入密码（输入时不会显示）
robot@192.168.1.100's password: ******

第 5 步：连接成功，进入 Linux 命令行
Welcome to Ubuntu 20.04.6 LTS (GNU/Linux 5.15.0-91-generic x86_64)

robot@robot-pc:~$
```

**重要提示**：
- ⚠️ **密码输入时不会显示**（连星号都没有），这是正常的
- ⚠️ **指纹确认只需第一次**，以后会自动信任
- ⚠️ 如果密码错误，会提示 `Permission denied`，重新输入即可

### 3.4 连接成功后的操作

连接成功后，你会看到 Linux 的命令提示符（如 `robot@robot-pc:~$`），现在可以执行 Linux 命令了：

```bash
# 查看当前目录
pwd
# 输出：/home/robot

# 查看文件列表
ls -la

# 进入项目目录
cd /home/robot/Zhipu_Pr_JAKA

# 查看项目文件
ls

# 运行 Python 脚本
python main_hand.py

# 查看系统信息
uname -a
cat /etc/os-release

# 查看磁盘空间
df -h

# 查看内存使用
free -h

# 查看运行的进程
ps aux | grep python

# 退出 SSH 连接
exit
```

---

## 四、SSH 进阶使用

### 4.1 SSH 密钥认证（免密码登录）

每次输入密码很麻烦，可以设置**密钥认证**，实现免密码登录。

#### 步骤 1：生成密钥对

在你的 Windows 电脑上执行：

```powershell
ssh-keygen -t rsa -b 4096
```

**执行过程**：
```
Generating public/private rsa key pair.
Enter file in which to save the key (C:\Users\YourName/.ssh/id_rsa): [直接回车]
Enter passphrase (empty for no passphrase): [直接回车，不设密码]
Enter same passphrase again: [直接回车]

Your identification has been saved in C:\Users\YourName\.ssh\id_rsa
Your public key has been saved in C:\Users\YourName\.ssh\id_rsa.pub
```

**生成的文件**：
- `id_rsa`：**私钥**（保密，不要给别人）
- `id_rsa.pub`：**公钥**（可以公开，要放到 Linux 上）

#### 步骤 2：将公钥复制到 Linux 电脑

**方法 1：使用 ssh-copy-id（如果 Windows 有这个命令）**

```powershell
ssh-copy-id robot@192.168.1.100
# 输入密码后，公钥会自动复制
```

**方法 2：手动复制（Windows 推荐）**

1. 在 Windows 上查看公钥内容：
```powershell
type $env:USERPROFILE\.ssh\id_rsa.pub
```

2. 复制输出的内容（一整行，以 `ssh-rsa` 开头）

3. SSH 登录到 Linux：
```powershell
ssh robot@192.168.1.100
```

4. 在 Linux 上创建 `.ssh` 目录并添加公钥：
```bash
mkdir -p ~/.ssh
echo "你的公钥内容" >> ~/.ssh/authorized_keys
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
exit
```

#### 步骤 3：测试免密码登录

```powershell
ssh robot@192.168.1.100
```

**应该直接登录，不需要输入密码！**

### 4.2 SSH 配置文件（简化连接）

创建 SSH 配置文件，可以**用简短的别名代替复杂的连接命令**。

#### 创建配置文件

在 PowerShell 中输入：

```powershell
notepad $env:USERPROFILE\.ssh\config
```

#### 添加配置内容

```
# 机器人 Linux 电脑
Host robot
    HostName 192.168.1.100
    User robot
    Port 22
    IdentityFile ~/.ssh/id_rsa

# 另一个设备示例（如无人机）
Host drone
    HostName 192.168.1.101
    User admin
    Port 22

# 实验室服务器
Host lab-server
    HostName 192.168.10.50
    User developer
    Port 2222
    IdentityFile ~/.ssh/id_rsa_lab
```

#### 使用简化命令

配置完成后，连接命令简化为：

```powershell
# 之前：ssh robot@192.168.1.100
# 现在：
ssh robot

# 之前：ssh -p 2222 developer@192.168.10.50
# 现在：
ssh lab-server
```

### 4.3 文件传输（SCP 和 SFTP）

#### SCP（Secure Copy）- 命令行文件传输

**从 Windows 上传文件到 Linux**：

```powershell
# 基本格式
scp 本地文件路径 robot@192.168.1.100:/远程路径

# 示例：上传 Python 脚本
scp C:\Projects\test.py robot@192.168.1.100:/home/robot/

# 上传整个文件夹
scp -r C:\Projects\myapp robot@192.168.1.100:/home/robot/
```

**从 Linux 下载文件到 Windows**：

```powershell
# 基本格式
scp robot@192.168.1.100:/远程文件路径 本地路径

# 示例：下载日志文件
scp robot@192.168.1.100:/home/robot/logs/app.log C:\Downloads\

# 下载整个文件夹
scp -r robot@192.168.1.100:/home/robot/project C:\Backups\
```

**使用配置文件别名**：

```powershell
# 如果配置了 SSH config
scp test.py robot:/home/robot/
scp -r robot:/home/robot/project C:\Backups\
```

#### SFTP（Secure File Transfer Protocol）- 交互式文件传输

**连接 SFTP**：

```powershell
sftp robot@192.168.1.100
```

**常用命令**：

```bash
# 查看远程文件
sftp> ls
sftp> ls -la

# 切换远程目录
sftp> cd /home/robot/project

# 切换本地目录
sftp> lcd C:\Downloads

# 下载文件
sftp> get file.txt
sftp> get logs/app.log

# 上传文件
sftp> put file.txt
sftp> put C:\Projects\test.py

# 下载整个文件夹
sftp> get -r project_folder

# 上传整个文件夹
sftp> put -r local_folder

# 退出
sftp> exit
```

### 4.4 端口转发（高级功能）

如果需要访问机器人上的 **Web 服务**（如 ROS2 Dashboard、摄像头画面）：

#### 本地端口转发

```powershell
# 将机器人上的 8080 端口映射到本地 8080 端口
ssh -L 8080:localhost:8080 robot@192.168.1.100
```

**然后在浏览器访问**：
```
http://localhost:8080
```

**实际应用场景**：

```powershell
# 访问机器人上的摄像头画面（假设在 8000 端口）
ssh -L 8000:localhost:8000 robot

# 访问 ROS2 Dashboard（假设在 8080 端口）
ssh -L 8080:localhost:8080 robot

# 访问其他服务
ssh -L 3000:localhost:3000 robot
```

**保持后台运行**：

```powershell
# 使用 -N 参数（不执行远程命令）-f 参数（后台运行）
ssh -N -f -L 8080:localhost:8080 robot
```

---

## 五、针对机器人项目的工作流程

### 5.1 典型开发流程

```powershell
# 1. 连接到机器人的 Linux 电脑
ssh robot@192.168.1.100

# 2. 进入项目目录
cd ~/Zhipu_Pr_JAKA

# 3. 激活 Python 虚拟环境（如果有）
source venv/bin/activate

# 4. 运行主程序
python main_hand.py

# 5. 或者运行持续对话系统
python continuous_dialogue.py

# 6. 查看日志
tail -f logs/app.log

# 7. 退出
exit
```

### 5.2 后台运行程序

如果程序需要**长时间运行**，使用以下方法：

#### 方法 1：nohup（简单后台运行）

```bash
# 后台运行程序，输出保存到 output.log
nohup python main_hand.py > output.log 2>&1 &

# 查看输出
tail -f output.log

# 查找进程
ps aux | grep main_hand

# 终止进程
kill <进程 ID>
```

#### 方法 2：screen（推荐，可恢复会话）

**安装 screen**（如果没有）：
```bash
sudo apt install screen
```

**创建新的 screen 会话**：
```bash
screen -S robot_app
```

**在 screen 中运行程序**：
```bash
python main_hand.py
```

**分离会话**（程序继续运行）：
- 按 `Ctrl + A`
- 然后按 `D`

**重新连接会话**：
```bash
screen -r robot_app
```

**查看所有会话**：
```bash
screen -ls
```

**终止会话**：
```bash
# 在 screen 内输入
exit

# 或从外部终止
screen -S robot_app -X quit
```

### 5.3 远程开发（VS Code Remote SSH）

使用 **VS Code 的 Remote SSH 插件**，可以直接在 Windows 上编辑 Linux 上的代码，**就像在本地一样**！

#### 安装插件

1. 打开 VS Code
2. 按 `Ctrl + Shift + X` 打开扩展面板
3. 搜索 **"Remote - SSH"**
4. 点击 **"安装"**

#### 配置连接

1. 按 `F1`，输入 `Remote-SSH: Connect to Host`
2. 选择 `robot`（如果配置了 config 文件）或输入 `robot@192.168.1.100`
3. 选择 Windows 上的 SSH 配置文件
4. 连接成功后，VS Code 会打开一个新窗口

#### 使用体验

- ✅ 可以直接编辑 Linux 上的文件
- ✅ 内置终端直接执行 Linux 命令
- ✅ 可以使用 VS Code 的所有功能（调试、Git 等）
- ✅ 文件保存后直接生效，无需上传

---

## 六、常见问题与解决方案

### 6.1 连接被拒绝

**错误信息**：
```
ssh: connect to host 192.168.1.100 port 22: Connection refused
```

**解决方案**：

1. **检查 SSH 服务是否运行**（在 Linux 上）：
```bash
sudo systemctl status ssh
```

2. **启动 SSH 服务**：
```bash
sudo systemctl start ssh
```

3. **设置开机自启**：
```bash
sudo systemctl enable ssh
```

4. **检查防火墙**：
```bash
# 开放 SSH 端口
sudo ufw allow 22
sudo ufw enable
```

### 6.2 找不到 IP 地址

**在 Linux 上查看 IP 地址**：

```bash
# 方法 1
ip addr show

# 方法 2
ifconfig

# 方法 3
hostname -I
```

**输出示例**：
```
inet 192.168.1.100/24 brd 192.168.1.255 scope global eth0
```

### 6.3 权限问题

**错误信息**：
```
Permissions 0644 for '/home/robot/.ssh/authorized_keys' are too open.
```

**解决方案**：

```bash
# 修复 .ssh 目录权限
chmod 700 ~/.ssh

# 修复 authorized_keys 权限
chmod 600 ~/.ssh/authorized_keys

# 修复用户目录权限
chmod 755 ~
```

### 6.4 密钥认证失败

**错误信息**：
```
Permission denied (publickey,password).
```

**解决方案**：

1. **检查公钥是否正确复制**：
```bash
cat ~/.ssh/authorized_keys
```

2. **检查权限**：
```bash
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
```

3. **检查 SSH 配置**（在 Linux 上）：
```bash
sudo nano /etc/ssh/sshd_config
```

确保以下配置正确：
```
PubkeyAuthentication yes
AuthorizedKeysFile .ssh/authorized_keys
```

4. **重启 SSH 服务**：
```bash
sudo systemctl restart ssh
```

### 6.5 中文乱码

**问题**：显示中文时出现乱码

**解决方案**：

1. **设置 Linux 终端编码**：
```bash
export LANG=zh_CN.UTF-8
export LC_ALL=zh_CN.UTF-8
```

2. **永久设置**（添加到 `~/.bashrc`）：
```bash
echo "export LANG=zh_CN.UTF-8" >> ~/.bashrc
echo "export LC_ALL=zh_CN.UTF-8" >> ~/.bashrc
source ~/.bashrc
```

### 6.6 断开连接

**问题**：SSH 连接经常自动断开

**解决方案**：

1. **在 Windows 的 SSH config 中添加**：
```
Host *
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

2. **或在连接时指定**：
```powershell
ssh -o ServerAliveInterval=60 robot@192.168.1.100
```

---

## 七、安全最佳实践

### 7.1 保护私钥

- 🔒 **不要分享私钥**（`id_rsa`）
- 🔒 **设置私钥密码**（生成时输入 passphrase）
- 🔒 **定期更换密钥**

### 7.2 使用强密码

- ✅ 至少 12 个字符
- ✅ 包含大小写字母、数字、特殊符号
- ✅ 不要使用常见单词

### 7.3 限制 SSH 访问

**在 Linux 上配置防火墙**：

```bash
# 只允许特定 IP 访问 SSH
sudo ufw allow from 192.168.1.0/24 to any port 22

# 拒绝其他所有访问
sudo ufw deny 22
```

### 7.4 修改默认端口（可选）

**在 Linux 上修改 SSH 配置**：

```bash
sudo nano /etc/ssh/sshd_config
```

修改端口：
```
Port 2222
```

重启服务：
```bash
sudo systemctl restart ssh
```

**连接时指定端口**：
```powershell
ssh -p 2222 robot@192.168.1.100
```

---

## 八、快速参考卡

### 8.1 常用命令速查

```powershell
# ============ 连接 ============
ssh user@ip                    # 基本连接
ssh -p port user@ip            # 指定端口
ssh user@ip "command"          # 执行单个命令
ssh -N -L 8080:localhost:8080  # 端口转发

# ============ 文件传输 ============
scp file user@ip:/path         # 上传文件
scp user@ip:/path/file .       # 下载文件
scp -r folder user@ip:/path    # 上传文件夹
sftp user@ip                   # 交互式传输

# ============ 密钥管理 ============
ssh-keygen -t rsa -b 4096      # 生成密钥
ssh-copy-id user@ip            # 复制公钥

# ============ 后台运行 ============
nohup command &                # 后台运行
screen -S name                 # 创建会话
screen -r name                 # 恢复会话
screen -ls                     # 查看会话
```

### 8.2 Linux 常用命令速查

```bash
# ============ 文件操作 ============
pwd                 # 显示当前目录
ls -la             # 列出文件
cd /path           # 切换目录
cp file1 file2     # 复制文件
mv file1 file2     # 移动文件
rm file            # 删除文件
mkdir dirname      # 创建目录

# ============ 查看文件 ============
cat file           # 查看文件内容
tail -f file       # 实时查看文件
less file          # 分页查看

# ============ 系统信息 ============
uname -a           # 系统信息
df -h              # 磁盘空间
free -h            # 内存使用
top                # 进程监控

# ============ 网络 ============
ip addr show       # 查看 IP
ping host          # 测试连通性
netstat -tulpn     # 查看端口

# ============ 进程管理 ============
ps aux             # 查看所有进程
kill PID           # 终止进程
pkill name         # 按名称终止进程

# ============ 其他 ============
exit               # 退出 SSH
Ctrl+C             # 终止当前命令
Ctrl+Z             # 挂起命令
bg                 # 后台运行挂起的命令
fg                 # 前台运行挂起的命令
```

---

## 九、实际应用场景示例

### 场景 1：部署新代码

```powershell
# 1. 在 Windows 上开发测试
# 在 VS Code 中编写代码...

# 2. 上传到 Linux
scp C:\Projects\new_feature.py robot:/home/robot/Zhipu_Pr_JAKA/voice/agents/

# 3. SSH 登录
ssh robot

# 4. 运行测试
cd ~/Zhipu_Pr_JAKA
python -m pytest tests/test_new_feature.py

# 5. 如果没问题，集成到主程序
# 编辑 main_hand.py，导入新模块...
```

### 场景 2：调试运行中的程序

```powershell
# 1. SSH 登录
ssh robot

# 2. 查看程序是否在运行
ps aux | grep main_hand

# 3. 查看日志
tail -f /home/robot/Zhipu_Pr_JAKA/logs/app.log

# 4. 如果程序卡住，终止并重启
pkill -f main_hand
cd ~/Zhipu_Pr_JAKA
python main_hand.py

# 5. 或者使用 screen 后台运行
screen -S robot_main
python main_hand.py
# Ctrl+A, D 分离会话
```

### 场景 3：多设备协同

```powershell
# 1. 同时连接多个设备
ssh robot      # 终端 1：机器人
ssh drone      # 终端 2：无人机
ssh lab-server # 终端 3：实验室服务器

# 2. 在设备间传输文件
scp robot:/home/robot/data.csv drone:/home/drone/

# 3. 使用端口转发访问服务
ssh -N -L 8080:robot:80 robot  # 终端 4
# 浏览器访问 localhost:8080
```

---

## 十、学习资源

### 10.1 官方文档

- [OpenSSH 官方文档](https://www.openssh.com/manual.html)
- [SSH.com 教程](https://www.ssh.com/academy/ssh)

### 10.2 进阶阅读

- 《Linux 命令行与 shell 脚本编程大全》
- 《UNIX 网络编程》
- 《SSH 权威指南》

### 10.3 在线工具

- [SSH 密钥生成器](https://sshkeygen.online/)
- [SSH 配置文件生成器](https://www.ssh.com/academy/ssh/config)

---

**文档结束**

---

## 附录 A：完整示例 - 从零开始配置

### A.1 第一次连接完整流程

```powershell
# 步骤 1：检查 SSH 是否安装
ssh -V
# 如果未安装，执行：
# Add-WindowsCapability -Online -Name OpenSSH.Client~~~~0.0.1.0

# 步骤 2：测试网络连通
ping 192.168.1.100

# 步骤 3：首次连接
ssh robot@192.168.1.100
# 输入 yes 确认指纹
# 输入密码

# 步骤 4：验证连接成功
pwd
ls -la

# 步骤 5：退出
exit
```

### A.2 配置免密码登录完整流程

```powershell
# 步骤 1：生成密钥
ssh-keygen -t rsa -b 4096
# 一路回车

# 步骤 2：查看公钥
type $env:USERPROFILE\.ssh\id_rsa.pub
# 复制输出的内容

# 步骤 3：登录 Linux
ssh robot@192.168.1.100

# 步骤 4：在 Linux 上配置
mkdir -p ~/.ssh
echo "你的公钥内容" >> ~/.ssh/authorized_keys
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
exit

# 步骤 5：测试免密码登录
ssh robot@192.168.1.100
# 应该直接登录
```

### A.3 配置 SSH config 完整流程

```powershell
# 步骤 1：创建配置文件
notepad $env:USERPROFILE\.ssh\config

# 步骤 2：添加内容
# Host robot
#     HostName 192.168.1.100
#     User robot
#     Port 22
#     IdentityFile ~/.ssh/id_rsa

# 步骤 3：保存并测试
ssh robot
# 应该直接连接
```

---

**祝你使用愉快！如有问题，欢迎反馈。** 🚀
