# Product Baseline V1

**版本**: product_baseline_v1  
**基线来源**: Stage 9-R2 Official Balanced Baseline v1  
**真实数据配置**: Stage 10-2R1 修正验证  
**状态**: ✅ **已冻结，作为上线候选版本**  
**日期**: 2026-04-20

---

## 1. 版本说明

### 1.1 为什么选这个基线

经过 Stage 9-10 完整实验验证，该基线满足以下标准：

1. **可复现**: 多次实验结果一致
2. **可解释**: 参数设计逻辑清晰
3. **经过反证**: R3 证明继续强化 L1 会破坏平衡
4. **外部有效**: S10-2R1 证明真实数据下有效

### 1.2 版本冻结声明

```
本版本自 2026-04-20 起冻结。
后续如需修改，必须：
1. 创建新版本 (v2, v3...)
2. 保留本版本完整备份
3. 新实验必须与本版本对比
```

---

## 2. 核心配置 (已冻结)

### 2.1 训练配置

```python
# ============================================
# Product Baseline V1 - 训练配置
# ============================================

# 采样策略
SAMPLING_RATIO = 'L1:L2:L3 = 2:2:1'  # 固定比例
STRICT_BALANCED_BATCH = True           # 严格平衡 batch

# 任务权重 (固定，不动态调整)
TASK_WEIGHTS = {
    'L1': 1.2,
    'L2': 1.2,
    'L3': 1.0,
}

# Replay 配置
REPLAY_INTERVAL = 10    # 每 10 步 replay L1
REPLAY_WEIGHT = 0.5     # replay 损失权重

# 课程学习阶段
PHASE_STEPS = {
    'L1_warmup': (0, 50),
    'L1_L2': (50, 120),
    'L1_L2_L3': (120, 800),
}

# 优化器配置
LEARNING_RATE = 1e-4
GRAD_CLIP = 1.0
OPTIMIZER = 'AdamW'

# 训练长度
NUM_STEPS = 800  # 产品版本标准长度
```

### 2.2 Output KL Guard 配置

```python
# ============================================
# Product Baseline V1 - Guard 配置
# ============================================

GUARD_CONFIG = {
    'beta': 0.2,           # KL 约束强度
    'use_probs': True,     # 使用概率分布
    'num_samples': 30,     # 锚点样本数
    'reference_capture': 'training_start',  # 基线捕获时机
}

# 监控阈值
GUARD_ALERT_THRESHOLD = 0.005  # Writeback Δ 超过此值告警
GUARD_BLOCK_THRESHOLD = 0.01   # Writeback Δ 超过此值阻断
```

### 2.3 TSLA 门控配置

```python
# ============================================
# Product Baseline V1 - TSLA 配置
# ============================================

TSLA_CONFIG = {
    'promotion_threshold': 0.8,    # 晋升到长期记忆阈值
    'isolation_threshold': 0.3,    # 隔离阈值
    'writeback_enabled': True,     # 写回开关
    'auto_promotion': False,       # 自动晋升 (产品版本关闭)
}

# 动作策略
TSLA_ACTIONS = {
    'instant_to_longterm': 'manual_review_required',  # 产品版本需人工审核
    'isolation': 'auto',                              # 自动隔离
    'writeback': 'guarded',                           # 受保护写回
}
```

### 2.4 真实数据配置

```python
# ============================================
# Product Baseline V1 - 真实数据配置
# ============================================

REAL_DATA_CONFIG = {
    'ratio': 0.15,           # 真实数据比例 15%
    'max_samples': 100,      # 每个级别最大样本数
    'source': 'squad_v2',    # 数据源
    'difficulty_mapping': {
        'L1': 'simple_qa',
        'L2': 'complex_qa',
        'L3': 'multi_hop_qa',
    }
}
```

---

## 3. 性能基准

### 3.1 训练表现基准

| 指标 | 基准值 | 验收标准 |
|------|--------|----------|
| 平衡窗口 | ≥ 399 steps | ≥ 200 steps |
| 后期平衡 (Step 500+) | ≥ 350 steps | ≥ 100 steps |
| 最终 L1 Accuracy | 50% | ≥ 40% |
| 最终 L2 Accuracy | 55% | ≥ 45% |
| 最终 L3 Accuracy | 40% | ≥ 30% |
| Writeback Max \|Δ\| | < 0.001 | < 0.005 |

### 3.2 稳定性基准

| 指标 | 基准值 | 验收标准 |
|------|--------|----------|
| L1 是否长期清零 | 否 | 否 |
| 后期是否崩塌 | 否 | 否 |
| Guard 是否持续工作 | 是 | 是 |
| TSLA 是否正常触发 | 是 | 是 |

---

## 4. 产品能力边界

### 4.1 可开放能力 ✅

适合优先上线/内测的能力：

1. **检索增强问答**
   - 基于文档的问答
   - 单层任务处理
   - 受控多层任务处理

2. **结构化任务处理**
   - 受控写回下的任务执行
   - 明确边界内的项目知识助手
   - 预定义工作流执行

3. **记忆辅助**
   - 短期记忆调用
   - 已验证知识的快速检索

### 4.2 谨慎开放能力 ⚠️

可以内测，但必须强监控：

1. **多层任务联动推理**
   - 需要实时监控平衡状态
   - 异常时自动降级

2. **长链路记忆调用**
   - 需要人工审核关键节点
   - 限制调用深度

3. **TSLA 分流参与的复杂任务**
   - 记录所有 TSLA 动作
   - 定期人工复核

### 4.3 暂不开放能力 ❌

现阶段不要直接开放给真实用户：

1. **高自由度自动晋升**
   - 原因: 未经充分验证
   - 策略: 人工审核所有晋升

2. **大范围自学习写回**
   - 原因: 风险过高
   - 策略: 仅开放受控写回

3. **未充分验证的复杂开放式多任务训练**
   - 原因: 超出当前验证范围
   - 策略: 限制任务类型和复杂度

---

## 5. 已证明不可取的配置

### 5.1 动态权重调整

```python
# ❌ 不可取
DYNAMIC_WEIGHT_ADJUSTMENT = True

# 原因: 调整速度跟不上模型学习速度
# 结果: 短暂平衡后完全切换到某一任务
```

### 5.2 过强 L1 保护

```python
# ❌ 不可取
TASK_WEIGHTS = {
    'L1': 1.3,  # 高于 1.2
    'L2': 1.2,
    'L3': 1.0,
}
REPLAY_INTERVAL = 8  # 高于每 10 步

# 原因: 破坏整体平衡
# 结果: L1/L2 同时下降，平衡窗口缩短
```

### 5.3 高比例真实数据

```python
# ❌ 不可取 (产品初期)
REAL_DATA_RATIO = 0.30  # 30%
NUM_STEPS = 500

# 原因: 系统需要更长时间适应
# 结果: 无法形成持续平衡窗口
# 修正: 15% + 800 steps 成功
```

---

## 6. 监控与告警

### 6.1 必须监控的指标

```python
MONITORING_METRICS = {
    # Writeback 监控
    'writeback_delta': {
        'alert_threshold': 0.005,
        'critical_threshold': 0.01,
        'action': 'log_alert / block_training',
    },
    
    # 任务平衡监控
    'task_balance': {
        'min_l1': 0.30,
        'min_l2': 0.30,
        'min_l3': 0.20,
        'action': 'alert / auto_degrade',
    },
    
    # TSLA 动作监控
    'tsla_actions': {
        'track_all': True,
        'review_promotions': True,
        'alert_isolation_rate': 0.1,  # 隔离率超过 10% 告警
    },
    
    # 错误率监控
    'error_rate': {
        'alert_threshold': 0.05,
        'critical_threshold': 0.10,
        'action': 'alert / pause_service',
    },
}
```

### 6.2 告警级别

| 级别 | 条件 | 动作 |
|------|------|------|
| INFO | 正常波动 | 记录日志 |
| WARN | 接近阈值 | 发送告警，人工关注 |
| ERROR | 超过阈值 | 自动降级，人工介入 |
| CRITICAL | 严重异常 | 阻断服务，紧急回滚 |

---

## 7. 回滚策略

### 7.1 自动回滚触发条件

```python
ROLLBACK_TRIGGERS = {
    'writeback_delta': {
        'threshold': 0.01,
        'duration': '5_minutes',
        'action': 'auto_rollback_to_baseline',
    },
    'task_balance': {
        'condition': 'l1 < 0.1 AND l2 < 0.1',
        'duration': '10_minutes',
        'action': 'auto_rollback_to_baseline',
    },
    'error_rate': {
        'threshold': 0.15,
        'duration': '5_minutes',
        'action': 'auto_rollback_to_baseline',
    },
}
```

### 7.2 手动回滚

```python
MANUAL_ROLLBACK = {
    'enabled': True,
    'authorization': 'admin_required',
    'rollback_target': 'product_baseline_v1',
    'rollback_time': '< 30_seconds',
}
```

---

## 8. 灰度策略

### 8.1 灰度阶段

| 阶段 | 用户范围 | 开放能力 | 监控强度 |
|------|----------|----------|----------|
| Phase 1 | 内部测试 (10人) | 可开放能力 | 全量监控 |
| Phase 2 | 小范围试点 (100人) | 可开放能力 | 全量监控 |
| Phase 3 | 扩大试点 (1000人) | 可开放 + 谨慎开放 | 全量监控 |
| Phase 4 | 全量上线 | 根据验证结果决定 | 标准监控 |

### 8.2 灰度准入标准

```python
GRAYSCALE_GATES = {
    'phase_1_to_2': {
        'min_duration': '1_week',
        'max_error_rate': 0.02,
        'no_critical_incidents': True,
    },
    'phase_2_to_3': {
        'min_duration': '2_weeks',
        'max_error_rate': 0.01,
        'user_satisfaction': '> 4.0',
    },
    'phase_3_to_4': {
        'min_duration': '1_month',
        'max_error_rate': 0.005,
        'stability_score': '> 0.95',
    },
}
```

---

## 9. 版本历史

| 版本 | 日期 | 说明 | 状态 |
|------|------|------|------|
| v0.1 (R1) | - | 动态权重实验 | ❌ 废弃 |
| v0.2 (R2) | - | 固定采样基线 | ✅ 演进为 v1 |
| v0.3 (R3) | - | 边界验证反例 | ⚠️ 归档 |
| **v1.0** | **2026-04-20** | **产品基线冻结** | **✅ 当前版本** |

---

## 10. 使用说明

### 10.1 如何复现

```bash
# 1. 准备数据
python prepare_data.py --baseline v1

# 2. 启动训练
python train.py --config product_baseline_v1.yaml

# 3. 验证结果
python validate.py --baseline v1 --report
```

### 10.2 如何扩展

如需创建新版本：

1. 复制本文件为 `PRODUCT_BASELINE_V2.md`
2. 明确记录所有变更
3. 进行对比实验
4. 性能必须不低于 v1

---

## 11. 联系与维护

- **版本维护**: ML Team
- **审批流程**: 任何修改需经技术负责人审批
- **变更记录**: 所有变更必须记录在本文件版本历史中

---

**版本**: product_baseline_v1  
**状态**: ✅ 已冻结  
**冻结日期**: 2026-04-20  
**下次评审**: 待定 (需经正式审批)
