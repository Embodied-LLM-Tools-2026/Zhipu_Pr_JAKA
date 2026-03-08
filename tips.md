# 团队 Git 协作指南

为了保证我们 5 个人在阅读和开发代码时互不干扰，并能随时同步师兄的最新优化，请大家严格遵守以下流程。

---

## 🛠 一、 初始环境搭建 (全员仅需执行一次)

如果你还没把代码弄到电脑上，请依次执行以下命令：

### 1. 克隆团队仓库到本地
打开终端（Terminal 或 Git Bash），进入你想存放代码的目录：
git clone [https://github.com/Embodied-LLM-Tools-2026/Zhipu_Pr_JAKA.git](https://github.com/Embodied-LLM-Tools-2026/Zhipu_Pr_JAKA.git)

### 2. 进入目录并使用 VS Code 打开
cd Zhipu_Pr_JAKA
code .

### 3. 连接师兄的原始仓库 (上游 upstream)
这一步是为了以后能一键拉取师兄那边的代码更新：
git remote add upstream [https://github.com/haitaoshen66/Zhipu_Pr_JAKA.git](https://github.com/haitaoshen66/Zhipu_Pr_JAKA.git)

### 4. 验证配置是否正确
git remote -v
预期结果：看到 origin 指向团队仓库，upstream 指向师兄仓库。

---

## 👨‍💻 二、 个人开发流程 (隔离开发，严禁在 main 直接修改)

### 1. 创建并切换到个人开发分支
建议命名格式：姓名缩写-dev（例如：qhy233-dev）。
git checkout -b <你的分支名>

### 2. 存档并提交本地修改
当你阅读完一段代码或改完一个模块后：
git add .
git commit -m "说明：这里填写你改了什么（例如：阅读核心逻辑并添加注释）"

### 3. 将你的分支推送到云端大本营
第一次推送新分支时执行：git push -u origin <你的分支名>
之后在该分支推送只需：git push

---

## 🔄 三、 如何合并与同步代码 (核心步骤)

### 1. 同步师兄的最新优化 (Upstream)
当师兄说“代码优化了，你们更新一下”时：
git checkout main            # 1. 先回到主分支
git fetch upstream           # 2. 抓取师兄的更新
git merge upstream/main      # 3. 把师兄的代码合进本地 main
git checkout <你的分支名>     # 4. 回到你自己的开发分支
git merge main               # 5. 让你的分支也用上师兄最新的代码

### 2. 同步队友的新功能 (Origin)
当队友的代码合并进我们组织的 main 之后：
git checkout main
git pull origin main         # 拉取大本营的最新汇总
git checkout <你的分支名>
git merge main               # 吸收队友的新功能

### 3. 提交你的代码给全组 (Pull Request)
当你觉得自己的部分写好了：
1. 打开浏览器进入 GitHub 组织仓库：Embodied-LLM-Tools-2026/Zhipu_Pr_JAKA
2. 点击醒目的绿色按钮 Compare & pull request。
3. 检查代码无误后提交。大家在微信群确认后，由负责人点击 Merge 合并。

---

## 💡 协作小贴士
* 频繁 Commit：建议每完成一个小功能就存一次档，不要憋大招。
* 冲突处理：如果 Merge 报错 Conflict，直接在 VS Code 界面点击冲突文件修改即可。
