# gpt2-chinese-poem-generator

> 基于 **GPT-2** 的中文古诗生成课程项目（PyTorch + Hugging Face Transformers）

---

## 1. 项目简介（Introduction）

本项目是一个面向高校人工智能/自然语言处理课程的完整实践工程，主题为**中文古典诗歌自动生成**。项目使用预训练语言模型 GPT-2，并在 `chinese-poetry` 数据集上进行微调（fine-tuning），实现从“输入关键词/起始句”到“自动生成古诗文本”的端到端流程。

项目不仅包含模型训练代码，还包括：
- 数据清洗与格式化脚本
- 训练与推理 Notebook
- 命令行生成工具
- Flask 网页演示系统

整体结构强调可复现、可展示、可报告，适用于课程大作业、中期汇报与毕业设计原型。

---

## 2. 项目目标（Objective）

本项目的核心目标如下：

1. **构建完整 NLP 生成式任务流程**：从原始 JSON 数据到可部署的文本生成 Demo。  
2. **掌握预训练模型微调方法**：以 GPT-2 为主线，理解 tokenizer、causal LM、Trainer 等组件。  
3. **提升中文古诗生成质量**：通过领域数据微调，使输出更符合古诗风格和语言节奏。  
4. **形成规范化科研工程**：代码结构清晰、实验步骤可重复、结果可追溯。

---

## 3. 方法（Method: GPT-2 + Fine-tuning）

### 3.1 模型框架
- 基础模型：`uer/gpt2-chinese-poem`
- 模型类型：Causal Language Model（自回归语言模型）
- 实现框架：PyTorch + Hugging Face Transformers

### 3.2 训练思路
1. 将清洗后的诗歌语料整理为“每行一首”。
2. 使用 GPT-2 tokenizer 对文本进行编码。
3. 采用 `Trainer` API 进行监督式语言模型微调。
4. 依据验证集 `eval_loss` 自动保存最佳模型（best checkpoint）。
5. 使用同一组 prompt 对比微调前后生成效果。

### 3.3 生成策略
推理时采用采样式生成（`do_sample=True`），并设置：
- `temperature`
- `top_p`
- `repetition_penalty`

以在“文本多样性”和“古诗可读性”之间取得平衡。

---

## 4. 数据集说明（chinese-poetry）

本项目使用开源中文古典诗词数据集 **`chinese-poetry`**（GitHub 上广泛使用的数据源）。

### 4.1 原始数据形式
- 多个 JSON 文件（如 `poet.tang.*.json`）
- 每条样本通常包含标题、作者、诗句列表（`paragraphs`）等字段

### 4.2 本项目的数据筛选规则
在 `data/process_data.py` 中，针对唐诗子集进行严格清洗：
- 必须恰好 **4 句**
- 每句必须恰好 **5 或 7 个中文字符**
- 用 `。` 将四句拼接为一行
- 全量遍历目录下所有 `poet.tang.*.json`
- 去重后写入训练文本文件

该规则有助于得到风格相对统一的训练语料，便于课程实验分析。

> 注：请在课程报告与公开发布时遵守数据集原始仓库的许可证与使用规范。

---

## 5. 项目结构

```text
gpt2-chinese-poem-generator/
│
├── data/
│   ├── raw/                         # 原始 chinese-poetry JSON 数据
│   ├── processed/                   # 清洗后语料（每行一首）
│   └── process_data.py              # 严格清洗脚本（4句、5/7字）
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb       # 数据清洗流程演示
│   └── 02_train_gpt2.ipynb          # GPT-2 微调与评估完整流程
│
├── app/
│   ├── app.py                       # Flask Web 演示入口
│   └── templates/
│       └── index.html               # Demo 页面模板
│
├── models/                          # 保存微调模型与 tokenizer
├── report/                          # 课程报告、图表、实验记录
│
├── src/
│   ├── prepare_data.py              # 通用清洗脚本（JSON->text）
│   ├── train.py                     # 训练脚本（Trainer API）
│   └── generate.py                  # 命令行生成脚本
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 6. 运行步骤（Step-by-Step）

> 以下步骤以 Linux/macOS 终端为例，Windows 可使用 PowerShell 并调整激活命令。

### Step 1. 克隆与进入项目
```bash
git clone <your-repo-url>
cd gpt2-chinese-poem-generator
```

### Step 2. 创建虚拟环境并安装依赖
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3. 准备原始数据
将 `chinese-poetry` 中相关 JSON 文件（如 `poet.tang.*.json`）放入：

```text
data/raw/
```

### Step 4. 数据清洗（严格规则）
```bash
python data/process_data.py \
  --input_dir data/raw \
  --output_file data/processed/tang_poems_4x5or7.txt
```

### Step 5. 训练 GPT-2（脚本方式）
```bash
python src/train.py \
  --train_file data/processed/tang_poems_4x5or7.txt \
  --model_name_or_path uer/gpt2-chinese-poem \
  --output_dir models/gpt2-poem-finetuned \
  --block_size 128 \
  --epochs 3 \
  --batch_size 8 \
  --learning_rate 5e-5 \
  --seed 42
```

### Step 6. 命令行生成测试
```bash
python src/generate.py \
  --model_dir models/gpt2-poem-finetuned \
  --prompt 春江花月夜 \
  --max_new_tokens 64
```

### Step 7. 启动 Flask 可视化 Demo
```bash
cd app
python app.py
```
浏览器打开：
```text
http://127.0.0.1:5000
```

### Step 8. Notebook 复现实验（可选）
- `notebooks/01_data_cleaning.ipynb`
- `notebooks/02_train_gpt2.ipynb`

---

## 7. 示例结果（Example Outputs）

以下为示例演示（仅作展示，实际结果随训练轮次、数据规模、随机种子而变化）：

### 输入 Prompt
- 春风又绿江南岸

### 微调前（示例）
- 春风又绿江南岸，旧梦天涯月满船。

### 微调后（示例）
- 春风又绿江南岸。烟柳轻摇入画栏。山色有无连远浦。渔歌一曲到前滩。

可以观察到：微调后结果在句式节奏与古诗风格上通常更加稳定。

---

## 8. 技术关键词（Keywords）

- **Transformer**
- **GPT-2**
- **NLP (Natural Language Processing)**
- **Text Generation**
- Fine-tuning
- Causal Language Modeling
- PyTorch
- Hugging Face Transformers

---

## 9. 课程项目建议（可用于答辩）

在 `report/` 中建议补充以下材料：
1. 模型与数据流程图
2. 超参数表格（batch size、learning rate、epoch、block size）
3. 训练/验证损失曲线
4. 微调前后案例对比与误差分析
5. 局限性与未来工作（押韵控制、格律约束、自动评价指标等）

---

## 10. 致谢与说明

- 感谢 `chinese-poetry` 数据贡献者与开源社区。
- 本项目为教学与科研练习用途，生成内容不代表事实陈述。

