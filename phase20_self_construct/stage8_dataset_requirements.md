# Stage 8 数据集需求清单

## 数据集类型与用途

### 1. 知识问答 (Knowledge QA) - 主要数据源

#### 推荐数据集

| 数据集名称 | 规模 | 语言 | 下载地址 | 适用层级 |
|-----------|------|------|----------|----------|
| **SQuAD 2.0** | 150K+ | 英文 | https://rajpurkar.github.io/SQuAD-explorer/ | L1-L2 |
| **Natural Questions** | 300K+ | 英文 | https://ai.google.com/research/NaturalQuestions | L1-L3 |
| **TriviaQA** | 650K+ | 英文 | http://nlp.cs.washington.edu/triviaqa/ | L1-L2 |
| **MS MARCO** | 1M+ | 英文 | https://microsoft.github.io/msmarco/ | L1-L3 |
| **DuReader** | 200K+ | 中文 | https://github.com/baidu/DuReader | L1-L3 |
| **CMRC 2018** | 20K+ | 中文 | https://github.com/ymcui/cmrc2018 | L1-L2 |

#### 建议选择

**英文场景**:
- 主数据集: **SQuAD 2.0** (质量高、格式标准)
- 补充: **Natural Questions** (真实查询、难度适中)

**中文场景**:
- 主数据集: **CMRC 2018** (学术标准、质量高)
- 补充: **DuReader** (真实场景、多样性)

---

### 2. 科学/常识推理 (Scientific/Common Sense Reasoning)

#### 推荐数据集

| 数据集名称 | 规模 | 类型 | 下载地址 | 适用层级 |
|-----------|------|------|----------|----------|
| **OpenBookQA** | 6K | 科学常识 | https://allenai.org/data/open-book-qa | L2-L3 |
| **ARC (AI2 Reasoning Challenge)** | 8K | 科学推理 | https://allenai.org/data/arc | L2-L3 |
| **CommonsenseQA** | 12K | 常识推理 | https://www.tau-nlp.org/commonsenseqa | L2-L3 |
| **QASC** | 10K | 多跳推理 | https://allenai.org/data/qasc | L3 |
| **Science QA** | 21K | 科学多模态 | https://scienceqa.github.io/ | L2-L3 |

#### 建议选择

- **OpenBookQA**: 适合 L2 级别，需要结合知识推理
- **ARC Challenge**: 适合 L3 级别，需要复杂推理

---

### 3. 结构化知识 (Structured Knowledge)

#### 知识图谱

| 资源名称 | 规模 | 类型 | 下载地址 |
|---------|------|------|----------|
| **ConceptNet** | 21M+ 关系 | 常识知识 | https://conceptnet.io/ |
| **Wikidata** | 1B+ 事实 | 百科知识 | https://www.wikidata.org/ |
| **WordNet** | 155K+ 词 | 词汇关系 | https://wordnet.princeton.edu/ |
| **ATOMIC** | 877K+ 推理 | 事件推理 | https://allenai.org/data/atomic-2020 |

#### 建议使用

- **ConceptNet**: 用于常识推理和知识扩展
- **ATOMIC**: 用于事件推理和因果推理

---

## 最小数据集 (MVP 版本)

### 快速启动所需

**目标**: 100 条 L1 级别样本验证完整流程

| 数据源 | 数量 | 用途 |
|--------|------|------|
| SQuAD 2.0 (简化) | 50 条 | 直接检索问答 |
| OpenBookQA (简化) | 30 条 | 简单推理 |
| 自建常识问题 | 20 条 | 系统适配测试 |

**总计**: 100 条

---

## 完整数据集 (正式版本)

### 训练集 (600 条)

| 难度 | 数量 | 数据源 |
|------|------|--------|
| L1 - 直接检索 | 300 条 | SQuAD + Natural Questions |
| L2 - 简单推理 | 200 条 | OpenBookQA + CommonsenseQA |
| L3 - 复杂推理 | 100 条 | ARC + QASC |

### 验证集 (100 条)

- 混合分布: L1 (50) + L2 (30) + L3 (20)
- 来源: 各数据集抽样，不重叠

### 测试集 (100 条)

- 混合分布: L1 (50) + L2 (30) + L3 (20)
- 来源: 独立数据集或人工构建

---

## 下载与处理脚本

### 快速下载命令

```bash
# 创建数据集目录
mkdir -p stage8_dataset/{train,val,test,raw}

# 1. SQuAD 2.0
cd stage8_dataset/raw
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json

# 2. OpenBookQA
git clone https://github.com/allenai/OpenBookQA.git

# 3. CommonsenseQA
git clone https://github.com/jonathanherzig/commonsenseqa.git

# 4. ARC
wget https://ai2-public-datasets.s3.amazonaws.com/arc/ARC-V1-Feb2018.zip
unzip ARC-V1-Feb2018.zip

# 5. ConceptNet
wget https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz
gunzip conceptnet-assertions-5.7.0.csv.gz
```

### 数据预处理

下载后需要运行预处理脚本将数据转换为 Stage 8 格式：

```python
# stage8_data_preprocessor.py
# 将原始数据集转换为统一格式
```

---

## 数据格式示例

### 转换后的样本格式

```json
{
    "id": "squad_001",
    "source": "SQuAD 2.0",
    "question": "What is the main product of photosynthesis?",
    "context": "Photosynthesis is the process by which plants convert light energy into chemical energy...",
    "difficulty": "L1",
    "domain": "biology",
    "expected_gap": 0,
    "expected_retrieval": 1,
    "expected_policy": 0,
    "answer": "glucose and oxygen",
    "explanation": "Photosynthesis converts carbon dioxide and water into glucose and oxygen using light energy",
    "knowledge_units": ["photosynthesis_definition", "photosynthesis_products"],
    "reasoning_chain": []
}
```

---

## 质量检查清单

下载后请确认：

- [ ] 数据集可以正常加载
- [ ] 样本数量符合预期
- [ ] 问题和答案格式正确
- [ ] 无明显噪声或错误
- [ ] 可以转换为 Stage 8 格式

---

## 下一步

1. **下载数据**: 根据上述清单下载所需数据集
2. **运行预处理**: 使用 `stage8_data_preprocessor.py` 转换格式
3. **质量检查**: 验证数据质量和格式
4. **开始训练**: 使用处理后的数据集进行 Stage 8 训练

---

**准备下载！建议先从 SQuAD 2.0 和 OpenBookQA 开始。**
