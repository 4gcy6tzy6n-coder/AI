# Stage 5 评估报告：受控自构建与参数晋升实验

## 概述

**阶段**: Stage 5 - 受控自构建与参数晋升实验  
**日期**: 2026-04-18  
**目标**: 验证框架能否在治理下真正自构建，并把少量高质量成果受控晋升

---

## 阶段 5 设计原则

### 红线（不可触碰）

1. **不允许候选内容跳过 TSLA、强审查、验证直接晋升**
2. **不允许真实用户偶发对话直接驱动参数晋升**
3. **不允许一开始就碰深层永久记忆**
4. **不允许自构建内容直接改写基础参数区**
5. **不允许为了提高通过率而降低治理门槛**

### 分阶段推进

| 子阶段 | 目标 | 产出 |
|--------|------|------|
| **Stage 5A** | 候选自构建 | 生成候选解释/关系/规则/模式 |
| **Stage 5B** | 知识库晋升 | 经 TSLA→强审查→验证→门控分流 |
| **Stage 5C** | 延迟参数晋升 | 极少量高质量候选进入参数区 |

---

## 核心文件

```
phase20_self_construct/
├── self_construct_candidate_runner.py    # Stage 5A: 候选生成
├── promotion_gate_runner.py              # Stage 5B: 晋升门控
├── self_learned_param_store_v1.py        # Stage 5C: 参数存储
├── promotion_rollback_tester.py          # 回滚测试
├── candidate_quality_benchmark.py        # 质量基准
├── candidates/                           # 候选存储
├── knowledge_base/                       # 知识库
│   ├── ephemeral.jsonl                   # 瞬时保留
│   ├── long_term_candidate.jsonl         # 长期候选
│   ├── isolation.jsonl                   # 隔离观察
│   └── error_archive.jsonl               # 错误归档
├── param_store/                          # 参数存储
│   ├── self_learned_params.pt            # 自学习参数
│   └── promotion_history.json            # 晋升历史
└── eval/                                 # 评估报告
    ├── stage5a_results.json
    ├── stage5b_results.json
    ├── stage5c_results.json
    ├── rollback_test_results.json
    └── quality_benchmark_results.json
```

---

## Stage 5A: 候选自构建实验

### 目标

让模型生成四类候选内容：
- **候选解释**: 对概念的解释、对决策的理由
- **候选关系**: 实体之间的关系
- **候选规则**: 如果 [条件]，则 [动作]
- **候选任务模式**: 完成某类任务的标准步骤

### 验收标准

| 标准 | 要求 | 机制 |
|------|------|------|
| 幻觉率低 | < 50% | 基于输出熵估计 |
| 结构合法 | > 90% | 字段完整性检查 |
| 有治理价值 | 有置信度 | 输出概率分布 |

### 核心实现

```python
class CandidateGenerator:
    def generate_candidate_explanation(self, query, context):
        # 1. 编码输入
        # 2. 获取模型输出
        # 3. 检查置信度
        # 4. 生成候选结构
        # 5. 幻觉检测
        # 6. 结构验证
        pass
```

### 幻觉检测机制

```python
def _estimate_hallucination(self, outputs):
    """基于输出概率分布的熵来估计幻觉"""
    gap_entropy = compute_entropy(outputs['gap_probs'])
    policy_entropy = compute_entropy(outputs['policy_probs'])
    hallucination = (gap_entropy + policy_entropy) / 2
    return hallucination
```

---

## Stage 5B: 知识库晋升实验

### 晋升流程

```
候选 → TSLA审查 → 强审查 → 验证 → 门控分流
         ↓           ↓         ↓         ↓
      八动作      结构/内容   模拟验证   五路分流
```

### TSLA 八动作审查

```python
ACTIONS = [
    "VERIFY_SCOPE",      # 验证范围
    "CHECK_PERMISSION",  # 检查权限
    "REJECT",           # 拒绝
    "LOG_INCIDENT",     # 记录事件
    "REQUEST_CLARIFICATION",  # 请求澄清
    "ESCALATE",         # 升级
    "ISOLATE",          # 隔离
    "ARCHIVE",          # 归档
]
```

### 门控分流决策

| 分数范围 | 决策 | 说明 |
|----------|------|------|
| ≥ 0.8 | 长期候选 | 高质量，可进入 Stage 5C |
| 0.5 - 0.8 | 瞬时保留 | 中等质量，短期保留 |
| 0.3 - 0.5 | 隔离观察 | 低质量，观察后决定 |
| < 0.3 | 错误归档 | 质量问题，归档备查 |

### 关键指标

- **晋升通过率**: 进入长期候选的比例（应 < 50%，precision 优先）
- **隔离率**: 进入隔离观察的比例
- **错误归档率**: 质量问题的比例（应 < 30%）

---

## Stage 5C: 延迟参数晋升实验

### 核心设计

```python
class SelfLearnedParamStore:
    def __init__(self, model):
        # 基础参数（冻结）
        self.base_params = {name: param.clone() 
                           for name, param in model.named_parameters()}
        
        # 自学习参数区（可更新）
        self.self_learned_params = {}
        
        # 晋升历史
        self.promotion_history = []
        
        # 回滚检查点
        self.rollback_checkpoints = []
```

### 晋升限制

- 每次最多晋升 **5** 个候选
- 最小晋升间隔 **100** 步
- 回滚阈值 **10%**（能力掉落超过 10% 触发回滚）

### 晋升映射

| 候选类型 | 目标模块 | 更新方式 | 幅度 |
|----------|----------|----------|------|
| EXPLANATION | gap_detector, policy_head | gradient_boost | 0.01 |
| RELATION | unit_encoder | embedding_adjust | 0.005 |
| RULE | governance_head, policy_head | bias_adjust | 0.008 |
| PATTERN | unit_encoder | pattern_enhance | 0.006 |

### 回滚机制

```python
def rollback(self, steps=1):
    """回滚到之前的状态"""
    checkpoint = self.rollback_checkpoints[-steps-1]
    
    for name, param in model.named_parameters():
        param.copy_(checkpoint['params'][name])
    
    self.promotion_count = checkpoint['promotion_count']
```

---

## 辅助工具

### 回滚测试器

测试晋升后：
1. 目标能力是否提升
2. 旧能力是否保持
3. 回滚机制是否有效

### 质量基准测试器

评估维度：
- **结构完整性** (30%): 字段是否完整
- **内容一致性** (30%): 类型与内容是否匹配
- **新颖性** (20%): 与已有候选的差异
- **抗幻觉** (20%): 输出确定性

---

## 阶段 5 验收标准

| 标准 | Stage 5A | Stage 5B | Stage 5C |
|------|----------|----------|----------|
| 候选生成质量稳定 | ✅ 结构 > 90% | - | - |
| 候选经治理后正确分流 | - | ✅ 晋升率 < 50% | - |
| 长期候选 precision 优先 | - | ✅ 错误归档 < 30% | - |
| 参数晋升后能力提升 | - | - | ✅ 目标能力 ↑ |
| 旧能力基本不掉 | - | - | ✅ 掉落 < 10% |
| 回滚机制真实可用 | - | - | ✅ 可恢复 |

---

## 运行方式

### 运行 Stage 5A

```bash
cd phase20_self_construct
python self_construct_candidate_runner.py
```

### 运行 Stage 5B

```bash
python promotion_gate_runner.py
```

### 运行 Stage 5C

```bash
python self_learned_param_store_v1.py
```

### 运行回滚测试

```bash
python promotion_rollback_tester.py
```

### 运行质量基准

```bash
python candidate_quality_benchmark.py
```

---

## 与阶段 4b 的关系

**阶段 4b** 证明了：
- Native 主干能训练
- Native 主干在充足数据下性能与 Transformer 相当

**阶段 5** 要证明：
- 这套框架不只是能学，还能在治理下真正"自我成长"
- 自构建内容可以被正确审查、分流、晋升
- 参数晋升是可控、可回滚的

---

## 下一步

阶段 5 完成后，可以进入：

**阶段 6**: 框架整合与系统测试
- 整合所有阶段成果
- 端到端系统测试
- 性能基准测试

**阶段 7**: 真实场景验证
- 小规模真实部署
- 持续监控与优化

---

## 总结

阶段 5 是整个框架中最具原创性的部分，它验证了新框架的核心假设：

> **AI 系统可以在严格治理下，自主生成知识并受控晋升到自身参数中。**

这不是无限制的"自我改进"，而是：
- **有审查**: TSLA + 强审查 + 验证
- **有分流**: 不是全部晋升，而是按质量分流
- **有回滚**: 晋升后持续监控，问题立即回滚
- **有限度**: 只晋升到自学习参数区，不动基础参数

这才是真正"受控"的自构建。
