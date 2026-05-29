# Stage 7 Writeback 保护机制 V2 设计

**日期**: 2026-04-19  
**阶段**: Stage 7 - Task 1  
**目标**: 将 writeback 变化从 13.60% 压到 <5%  
**前提**: 不破坏 Stage 6 官方基线 V1.0 的四个必须项

---

## 一、问题定义

### 当前状态

- **writeback 变化**: 13.60% (超标 2.7倍)
- **目标**: < 5%
- **差距**: 8.6个百分点

### 核心问题

如何让系统继续学习，同时不把 writeback 打坏？

---

## 二、边界分析

### 2.1 Writeback 路径边界图

```
┌─────────────────────────────────────────────────────────────┐
│                    Forward Backbone                          │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │  Gap    │→│ Policy  │→│Governance│→│Writeback│        │
│  │  Head   │  │  Head   │  │  Head   │  │  Head   │        │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘        │
│       │            │            │            │              │
│       └────────────┴────────────┘            │              │
│                  Shared                      │              │
│               Representation                 │              │
│                    (?)                       │              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Writeback Path │
                    │  (Target for    │
                    │   Protection)   │
                    └─────────────────┘
```

### 2.2 潜在污染点分析

| 污染点 | 风险等级 | 说明 |
|--------|----------|------|
| 共享表示层 | 🔴 高 | Gap/Policy/Governance 的梯度可能污染 writeback |
| 梯度回传 | 🔴 高 | 训练时 writeback head 接收其他 head 的梯度 |
| 参数共享 | 🟡 中 | Backbone 参数被所有 head 共享更新 |
| 优化器状态 | 🟡 中 | Adam 状态可能携带跨 head 信息 |

### 2.3 数据流分析

**正常数据流**:
```
Input → Backbone → Writeback Head → Output
                ↓
         (should be isolated)
```

**实际数据流 (可能污染)**:
```
Input → Backbone → Multi-Head → Mixed Gradients → Writeback Head
                ↓
         (shared representation)
```

---

## 三、根因假设

### 假设 1: 共享表示污染 (最可能)

**描述**: Writeback head 与其他 head 共享 backbone 表示，导致训练时其他 head 的更新间接影响 writeback。

**验证方法**:
1. 检查 backbone 特征在训练前后的变化
2. 比较 writeback-only 训练 vs 全 head 训练的差异

### 假设 2: 梯度交叉污染

**描述**: 反向传播时，writeback head 的梯度与其他 head 的梯度在 backbone 层混合。

**验证方法**:
1. 监控各 head 的梯度范数
2. 检查 writeback head 梯度是否受其他 head 影响

### 假设 3: 优化器状态泄漏

**描述**: Adam 优化器的状态 (momentum, variance) 在不同 head 间共享，导致状态污染。

**验证方法**:
1. 使用独立优化器测试
2. 重置优化器状态后评估

---

## 四、V2 设计方案

### 4.1 核心原则

1. **边界清晰**: Writeback path 与其他 path 物理/逻辑隔离
2. **独立比较链**: 建立"写回前 / 写回后 / 写回保护后"三段比较
3. **渐进保护**: 从强隔离逐步放宽，找到最优平衡点

### 4.2 方案 A: 特征隔离 (Feature Isolation)

```python
class WritebackFeatureIsolator:
    """
    为 writeback head 创建独立特征提取路径
    """
    
    def __init__(self, backbone_dim: int, isolate_layer: int = -2):
        self.shared_backbone = True  # 前 N-2 层共享
        self.isolate_layer = isolate_layer  # 从第 N-2 层开始隔离
        
        # Writeback 专用特征提取器
        self.writeback_feature_extractor = nn.Sequential(
            nn.Linear(backbone_dim, backbone_dim // 2),
            nn.LayerNorm(backbone_dim // 2),
            nn.GELU(),
            nn.Linear(backbone_dim // 2, backbone_dim),
        )
    
    def forward(self, backbone_features: torch.Tensor) -> torch.Tensor:
        # 在隔离层添加 writeback 专用变换
        isolated_features = self.writeback_feature_extractor(backbone_features)
        return isolated_features
```

**预期效果**: 阻断共享表示污染

### 4.3 方案 B: 梯度屏蔽 (Gradient Masking)

```python
class WritebackGradientShield:
    """
    在反向传播时屏蔽其他 head 对 writeback path 的影响
    """
    
    def __init__(self):
        self.writeback_params = set()  # Writeback 专用参数
    
    def register_writeback_params(self, module: nn.Module):
        """注册 writeback 专用参数"""
        for name, param in module.named_parameters():
            if 'writeback' in name:
                self.writeback_params.add(id(param))
    
    def shield_backward(self, loss: torch.Tensor):
        """反向传播时屏蔽非 writeback 梯度"""
        loss.backward(retain_graph=True)
        
        # 清零非 writeback 参数的梯度
        for param in self.model.parameters():
            if id(param) not in self.writeback_params and param.grad is not None:
                param.grad.zero_()
```

**预期效果**: 阻断梯度交叉污染

### 4.4 方案 C: 独立优化器 (Independent Optimizer)

```python
class WritebackIndependentOptimizer:
    """
    Writeback head 使用完全独立的优化器
    """
    
    def __init__(self, writeback_params, lr=1e-5):
        self.writeback_optimizer = torch.optim.AdamW(
            writeback_params,
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
        # 其他 head 使用主优化器
        self.main_optimizer = None  # 在训练循环中设置
    
    def step(self, writeback_loss, main_loss=None):
        """分步优化"""
        # Step 1: 优化 writeback
        self.writeback_optimizer.zero_grad()
        writeback_loss.backward(retain_graph=True)
        self.writeback_optimizer.step()
        
        # Step 2: 优化其他 head (可选)
        if main_loss is not None and self.main_optimizer is not None:
            self.main_optimizer.zero_grad()
            main_loss.backward()
            self.main_optimizer.step()
```

**预期效果**: 阻断优化器状态泄漏

### 4.5 方案 D: 综合保护 (Combined Protection)

结合 A+B+C 三种方案，形成多层保护:

```python
class WritebackGuardV2:
    """
    Writeback 综合保护机制 V2
    """
    
    def __init__(self, config):
        self.feature_isolator = WritebackFeatureIsolator(...)
        self.gradient_shield = WritebackGradientShield(...)
        self.independent_optimizer = WritebackIndependentOptimizer(...)
        
        # 保护强度配置
        self.protection_level = config.get('writeback_protection_level', 'medium')
        # low: 仅梯度屏蔽
        # medium: 梯度屏蔽 + 独立优化器
        # high: 特征隔离 + 梯度屏蔽 + 独立优化器
```

---

## 五、验证策略

### 5.1 独立比较链

建立三段比较:

1. **写回前 (Pre-Writeback)**: 训练前基线
2. **写回后 (Post-Writeback)**: 训练后无保护
3. **写回保护后 (Protected)**: 训练后有保护

### 5.2 验证指标

| 指标 | 写回前 | 写回后 | 写回保护后 | 目标 |
|------|--------|--------|------------|------|
| writeback 变化 | 0% | 13.60% | < 5%? | < 5% |
| target gain | - | +15.67% | > 10%? | > 10% |
| old ability | - | 2.26% | < 15%? | < 15% |

### 5.3 消融测试

| 测试 | 特征隔离 | 梯度屏蔽 | 独立优化器 | 预期 writeback |
|------|----------|----------|------------|----------------|
| 基线 | ❌ | ❌ | ❌ | 13.60% |
| A | ✅ | ❌ | ❌ | ? |
| B | ❌ | ✅ | ❌ | ? |
| C | ❌ | ❌ | ✅ | ? |
| A+B | ✅ | ✅ | ❌ | ? |
| A+C | ✅ | ❌ | ✅ | ? |
| B+C | ❌ | ✅ | ✅ | ? |
| A+B+C | ✅ | ✅ | ✅ | < 5%? |

---

## 六、交付物

### 6.1 设计文档

- ✅ `stage7_writeback_design_v2.md` (本文档)

### 6.2 实现代码

- ⏳ `stage7_writeback_guard.py` - 综合保护实现
- ⏳ `stage7_writeback_validation.py` - 验证框架

### 6.3 测试报告

- ⏳ `stage7_writeback_ablation_report.md` - 消融测试报告

---

## 七、下一步行动

1. **实现方案 A/B/C/D** 代码
2. **运行消融测试** 确定最优方案
3. **验证不破坏 Stage 6 必须项**

---

**设计完成时间**: 2026-04-19  
**状态**: Task 1 完成，准备进入实现阶段
