# 前沿论文下载指南

> **创建日期**：2026-03-16  
> **用途**：大创项目前沿技术与论文调研

---

## 一、核心论文列表

### 1. Code as Policies: Language Model Programs for Embodied Control

**论文信息**：
- **作者**：Jacky Liang, Wenlong Huang, et al. (Google Robotics)
- **发表时间**：2022年9月
- **会议**：ICRA 2023
- **arXiv ID**：2209.07753

**下载链接**：
- 📄 **PDF**: https://arxiv.org/pdf/2209.07753.pdf
- 🌐 **arXiv 页面**: https://arxiv.org/abs/2209.07753
- 💻 **代码**: https://code-as-policies.github.io/

**核心贡献**：
- 通过 LLM 生成 Python 代码作为机器人策略
- 实现了从自然语言到可执行代码的转换
- 在真实机器人上验证了方法的有效性

---

### 2. SayCan: Do Large Language Models Know What to Do Next?

**论文信息**：
- **作者**：Michael Ahn, Anthony Brohan, et al. (Google)
- **发表时间**：2022年7月
- **会议**：CoRL 2022
- **arXiv ID**：2207.05608

**下载链接**：
- 📄 **PDF**: https://arxiv.org/pdf/2207.05608.pdf
- 🌐 **arXiv 页面**: https://arxiv.org/abs/2207.05608
- 💻 **项目页面**: https://say-can.github.io/

**核心贡献**：
- 提出 LLM + Affordance 的任务规划框架
- LLM 负责"说什么"（任务规划）
- 价值函数负责"能做什么"（可行性）
- 在真实机器人上验证长程任务

---

### 3. RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control

**论文信息**：
- **作者**：Anthony Brohan, Noah Brown, et al. (Google DeepMind)
- **发表时间**：2023年7月
- **会议**：-
- **arXiv ID**：2307.15818

**下载链接**：
- 📄 **PDF**: https://arxiv.org/pdf/2307.15818.pdf
- 🌐 **arXiv 页面**: https://arxiv.org/abs/2307.15818
- 💻 **项目页面**: https://robotics-transformer2.github.io/

**核心贡献**：
- 将机器人控制转化为"下一个 token 预测"
- 视觉-语言-动作（VLA）模型
- 直接从互联网数据学习通用技能
- 泛化能力显著提升

---

### 4. VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models

**论文信息**：
- **作者**：Wenlong Huang, Chen Wang, et al. (Stanford, Li Fei-Fei 团队)
- **发表时间**：2023年
- **会议**：CoRL 2023
- **arXiv ID**：2309.12291

**下载链接**：
- 📄 **PDF**: https://arxiv.org/pdf/2309.12291.pdf
- 🌐 **arXiv 页面**: https://arxiv.org/abs/2309.12291
- 💻 **项目页面**: https://voxposer.github.io/

**核心贡献**：
- 通过基础模型生成 3D 轨迹值图
- 组合多个 VLM 和 LLM 的推理能力
- 无需训练即可泛化到新场景
- Zero-shot 机器人操作

---

### 5. ProgPrompt: Generating Situated Robot Task Plans using Large Language Models

**论文信息**：
- **作者**：Ishika Singh, Gargi Singh, et al. (USC, NVIDIA)
- **发表时间**：2022年9月
- **会议**：ICRA 2023
- **arXiv ID**：2209.11302

**下载链接**：
- 📄 **PDF**: https://arxiv.org/pdf/2209.11302.pdf
- 🌐 **arXiv 页面**: https://arxiv.org/abs/2209.11302
- 💻 **代码**: https://github.com/NVlabs/progprompt

**核心贡献**：
- 通过自然语言提示生成机器人程序
- 提供详细的 API 文档作为上下文
- 生成可解释、可调试的代码
- 在仿真和真实环境中验证

---

## 二、扩展论文列表

### 6. Inner Monologue: Embodied Reasoning through Planning with Language Models

**论文信息**：
- **作者**：Wenlong Huang, Fei Xia, et al. (Google)
- **发表时间**：2022年
- **arXiv ID**：2207.05608

**下载链接**：
- 📄 **PDF**: https://arxiv.org/pdf/2207.05608.pdf
- 🌐 **arXiv 页面**: https://arxiv.org/abs/2207.05608

**核心贡献**：
- 内心独白式的推理框架
- 多轮对话式任务规划

---

### 7. Socratic Models: Composing Zero-Shot Multimodal Reasoning with Language

**论文信息**：
- **作者**：Andy Zeng, Maria Attarian, et al. (Google)
- **发表时间**：2022年
- **arXiv ID**：2204.00598

**下载链接**：
- 📄 **PDF**: https://arxiv.org/pdf/2204.00598.pdf
- 🌐 **arXiv 页面**: https://arxiv.org/abs/2204.00598

**核心贡献**：
- 多模态推理框架
- 零样本组合推理

---

### 8. PaLM-E: An Embodied Multimodal Language Model

**论文信息**：
- **作者**：Danny Driess, Fei Xia, et al. (Google)
- **发表时间**：2023年
- **arXiv ID**：2303.03378

**下载链接**：
- 📄 **PDF**: https://arxiv.org/pdf/2303.03378.pdf
- 🌐 **arXiv 页面**: https://arxiv.org/abs/2303.03378
- 💻 **项目页面**: https://palm-e.github.io/

**核心贡献**：
- 具身多模态语言模型
- 将机器人感知融入语言模型

---

## 三、下载方法

### 方法 1：直接下载 PDF

点击上述每个论文的 **PDF 链接**，浏览器会自动下载。

### 方法 2：使用命令行下载（推荐）

在 PowerShell 中使用 `curl` 或 `wget` 批量下载：

```powershell
# 创建论文存储目录
mkdir C:\Papers\EmbodiedAI

# 下载所有论文
cd C:\Papers\EmbodiedAI

# 下载 Code as Policies
curl -o "01_Code_as_Policies.pdf" https://arxiv.org/pdf/2209.07753.pdf

# 下载 SayCan
curl -o "02_SayCan.pdf" https://arxiv.org/pdf/2207.05608.pdf

# 下载 RT-2
curl -o "03_RT-2.pdf" https://arxiv.org/pdf/2307.15818.pdf

# 下载 VoxPoser
curl -o "04_VoxPoser.pdf" https://arxiv.org/pdf/2309.12291.pdf

# 下载 ProgPrompt
curl -o "05_ProgPrompt.pdf" https://arxiv.org/pdf/2209.11302.pdf

# 下载 PaLM-E
curl -o "06_PaLM-E.pdf" https://arxiv.org/pdf/2303.03378.pdf

# 下载 Socratic Models
curl -o "07_Socratic_Models.pdf" https://arxiv.org/pdf/2204.00598.pdf
```

### 方法 3：使用 Python 批量下载

创建 Python 脚本 `download_papers.py`：

```python
import requests
import os

papers = [
    {
        "title": "Code as Policies",
        "arxiv_id": "2209.07753",
        "filename": "01_Code_as_Policies.pdf"
    },
    {
        "title": "SayCan",
        "arxiv_id": "2207.05608",
        "filename": "02_SayCan.pdf"
    },
    {
        "title": "RT-2",
        "arxiv_id": "2307.15818",
        "filename": "03_RT-2.pdf"
    },
    {
        "title": "VoxPoser",
        "arxiv_id": "2309.12291",
        "filename": "04_VoxPoser.pdf"
    },
    {
        "title": "ProgPrompt",
        "arxiv_id": "2209.11302",
        "filename": "05_ProgPrompt.pdf"
    },
    {
        "title": "PaLM-E",
        "arxiv_id": "2303.03378",
        "filename": "06_PaLM-E.pdf"
    },
    {
        "title": "Socratic Models",
        "arxiv_id": "2204.00598",
        "filename": "07_Socratic_Models.pdf"
    }
]

# 创建下载目录
download_dir = "C:\\Papers\\EmbodiedAI"
os.makedirs(download_dir, exist_ok=True)

# 下载论文
for paper in papers:
    url = f"https://arxiv.org/pdf/{paper['arxiv_id']}.pdf"
    filepath = os.path.join(download_dir, paper['filename'])
    
    print(f"正在下载: {paper['title']}...")
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            f.write(response.content)
        
        print(f"✅ 下载完成: {paper['filename']}")
    except Exception as e:
        print(f"❌ 下载失败: {paper['title']} - {e}")

print("\n所有论文下载完成！")
```

运行脚本：

```powershell
python download_papers.py
```

---

## 四、论文阅读建议

### 4.1 阅读顺序

**第一阶段：基础理解**（1-2 周）
1. **SayCan** - 理解 LLM + Affordance 的基本框架
2. **ProgPrompt** - 学习提示词工程在机器人中的应用
3. **Code as Policies** - 理解代码生成的核心思想

**第二阶段：深入理解**（2-3 周）
4. **RT-2** - 学习 VLA 模型的架构
5. **VoxPoser** - 理解 3D 值图和零样本泛化
6. **PaLM-E** - 学习多模态语言模型

**第三阶段：扩展阅读**（1-2 周）
7. **Socratic Models** - 理解多模态推理框架
8. 其他相关论文（根据兴趣选择）

### 4.2 阅读方法

**快速浏览**（15-30 分钟）：
1. 阅读 **Abstract** 和 **Introduction**
2. 查看 **Figures** 和 **Tables**
3. 阅读 **Conclusion**
4. 决定是否深入阅读

**深入阅读**（2-4 小时）：
1. 仔细阅读 **Method** 部分
2. 理解 **Experiments** 设计
3. 查看 **Code** 和 **Demo**（如果有）
4. 记录关键思想和可借鉴之处

**笔记模板**：

```markdown
# 论文笔记：[论文标题]

## 基本信息
- 标题：
- 作者：
- 发表时间：
- 会议/期刊：
- arXiv ID：

## 核心问题
- 解决了什么问题？

## 核心方法
- 提出了什么方法？
- 关键创新点是什么？

## 实验结果
- 在什么数据集上验证？
- 性能如何？
- 有什么局限性？

## 与我的项目的关系
- 可以借鉴什么？
- 如何应用到我的项目中？

## 关键引用
- 引用了哪些重要论文？

## 我的想法
- 有什么改进思路？
- 有什么疑问？
```

---

## 五、相关资源

### 5.1 论文整理网站

- **arXiv Sanity**: https://arxiv-sanity-lite.com/
- **Papers with Code**: https://paperswithcode.com/
- **Semantic Scholar**: https://www.semanticscholar.org/

### 5.2 具身智能论文合集

- **Awesome LLM Robotics**: https://github.com/GT-RIPL/Awesome-LLM-Robotics
- **Embodied AI Papers**: https://github.com/CH-CH-CH/awesome-embodied-AI

### 5.3 学术搜索

- **Google Scholar**: https://scholar.google.com/
- **DBLP**: https://dblp.org/
- **Connected Papers**: https://www.connectedpapers.com/

---

## 六、论文管理工具

### 6.1 推荐工具

1. **Zotero**（免费，推荐）
   - 自动提取论文信息
   - 支持多种引用格式
   - 云同步

2. **Mendeley**（免费）
   - PDF 阅读和标注
   - 文献管理
   - 协作功能

3. **EndNote**（付费）
   - 功能强大
   - 与 Word 集成

### 6.2 Zotero 快速上手

1. 下载安装：https://www.zotero.org/
2. 安装浏览器插件
3. 在 arXiv 页面点击插件图标，自动保存论文
4. 使用标签分类管理

---

**文档结束**
