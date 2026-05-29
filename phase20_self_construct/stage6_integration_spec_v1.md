# Stage 6 全链路整合规范 v1.0

**日期**: 2026-04-18  
**阶段**: Stage 6.1 - 全链路整合  
**目标**: 将已验证的受控自构建机制整合为统一运行主链

---

## 1. 系统架构概览

### 1.1 全链路数据流

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Stage 6 全链路整合系统                               │
└─────────────────────────────────────────────────────────────────────────────┘

输入: 用户查询 / 系统触发
    │
    ▼
┌─────────────────┐
│  Native Backbone │  ← 原生主干 (phase19)
│  (Tiny v1)       │
└────────┬────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                    多头输出层                            │
│  ┌─────────┐  ┌──────────┐  ┌─────────────┐  ┌────────┐ │
│  │ Gap     │  │ Policy   │  │ Governance  │  │Writeback│ │
│  │ (0/1)   │  │(0/1/2)   │  │  (安全/合规) │  │(0/1)   │ │
│  └────┬────┘  └────┬─────┘  └──────┬──────┘  └───┬────┘ │
└───────┼────────────┼───────────────┼─────────────┼──────┘
        │            │               │             │
        ▼            ▼               ▼             ▼
   触发候选生成   路由决策        安全检查        记忆更新
        │            │               │             │
        └────────────┴───────┬───────┴─────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │      候选生成器 (Candidate    │
              │      Generator)               │
              │  - 从交互中提取潜在知识        │
              │  - 生成 RELATION/EXPLANATION  │
              │  - 质量评分与元数据            │
              └──────────────┬───────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │      类型分流器 (Type Router) │
              │                               │
              │  输入: 候选 + 质量分数         │
              │  决策:                         │
              │    score ≥ 0.90 + RELATION    │
              │      → 参数晋升通道            │
              │    score ≥ 0.80 + EXPLAIN/RULE│
              │      → 知识库通道              │
              │    score < 0.80               │
              │      → 丢弃或长期候选池        │
              └──────────────┬───────────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
            ▼                │                ▼
┌─────────────────────┐     │    ┌──────────────────────┐
│   参数晋升模块        │     │    │   知识库晋升模块      │
│   (Param Promotion) │     │    │   (KB Promotion)     │
│                     │     │    │                      │
│  配置: 6B 默认        │     │    │  - 存入向量数据库     │
│  - KL: 0.30/0.38     │     │    │  - 建立索引          │
│  - LR: 1e-5          │     │    │  - 可检索            │
│  - Replay: 35%       │     │    │                      │
│  - Step1 cap: 0.003  │     │    │                      │
│                     │     │    │                      │
│  流程:               │     │    │                      │
│  1. 生成训练样本     │     │    │                      │
│  2. 混合旧能力回放   │     │    │                      │
│  3. 分层 KL 约束     │     │    │                      │
│  4. 自适应调整       │     │    │                      │
│  5. 每步评估         │     │    │                      │
└──────────┬──────────┘     │    └──────────────────────┘
           │                │
           └────────────────┘
                          │
                          ▼
           ┌──────────────────────────────┐
           │      回滚钩子 (Rollback Hook) │
           │                               │
           │  - 保存晋升前基线              │
           │  - 每步后评估                  │
           │  - 超标时触发回滚              │
           │  - 恢复原始参数                │
           └──────────────────────────────┘
                          │
                          ▼
           ┌──────────────────────────────┐
           │      统一指标收集器            │
           │  (Unified Metrics Collector)  │
           │                               │
           │  - 每步目标提升               │
           │  - 每步旧能力掉落             │
           │  - writeback 变化             │
           │  - rollback 结果              │
           │  - 累积效应                   │
           └──────────────────────────────┘
```

### 1.2 模块连接关系

```python
# 主链调用顺序
class Stage6Orchestrator:
    def run_promotion_cycle(self, input_data):
        # 1. 主干推理
        backbone_outputs = self.backbone(input_data)
        
        # 2. 多头决策
        gap_decision = backbone_outputs['gap']
        policy_route = backbone_outputs['policy']
        governance_check = backbone_outputs['governance']
        
        # 3. 候选生成
        if gap_decision == 1:  # 需要新知识
            candidates = self.candidate_generator.generate(
                input_data, 
                backbone_outputs
            )
            
            # 4. 类型分流
            for candidate in candidates:
                route = self.type_router.route(candidate)
                
                if route == 'PARAM_PROMOTION':
                    # 5a. 参数晋升
                    result = self.param_promoter.promote(
                        candidate,
                        config=STAGE6_DEFAULT_CONFIG
                    )
                    
                    # 6. 回滚检查
                    if result['old_ability_drop'] > THRESHOLD:
                        self.rollback_hook.rollback()
                        
                elif route == 'KB_PROMOTION':
                    # 5b. 知识库晋升
                    self.kb_promoter.promote(candidate)
                    
        # 7. 指标收集
        metrics = self.metrics_collector.collect()
        return metrics
```

---

## 2. 模块接口定义

### 2.1 Native Backbone (已有)

```python
# 输入
Input: torch.Tensor [batch_size, seq_len]

# 输出
Output: {
    'gap_logits': Tensor [batch, 2],
    'gap_probs': Tensor [batch, 2],
    'policy_logits': Tensor [batch, 3],
    'policy_probs': Tensor [batch, 3],
    'governance_logits': Tensor [batch, 2],
    'governance_probs': Tensor [batch, 2],
    'writeback_logits': Tensor [batch, 2],
    'writeback_probs': Tensor [batch, 2],
}
```

### 2.2 Candidate Generator (整合)

```python
class CandidateGenerator:
    """候选生成器"""
    
    def generate(
        self,
        input_context: str,
        backbone_outputs: Dict,
        num_candidates: int = 5
    ) -> List[Candidate]:
        """
        从交互上下文生成候选
        
        Returns:
            List[Candidate]: 候选列表，每个包含:
                - candidate_id: str
                - candidate_type: RELATION/EXPLANATION/RULE/PATTERN
                - content: str
                - entities: Dict
                - metadata: {quality_score, source, timestamp}
        """
        pass
```

### 2.3 Type Router (整合)

```python
class TypeRouter:
    """类型分流器"""
    
    # 分流规则
    RULES = {
        'PARAM_PROMOTION': {
            'types': ['RELATION'],
            'min_score': 0.90,
            'mappable': True,
        },
        'KB_PROMOTION': {
            'types': ['EXPLANATION', 'RULE'],
            'min_score': 0.80,
        },
        'LONG_TERM': {
            'types': ['ALL'],
            'min_score': 0.70,
        },
        'DISCARD': {
            'types': ['ALL'],
            'min_score': 0.0,
        }
    }
    
    def route(self, candidate: Candidate) -> str:
        """
        决定候选的晋升路径
        
        Returns:
            'PARAM_PROMOTION' | 'KB_PROMOTION' | 'LONG_TERM' | 'DISCARD'
        """
        pass
```

### 2.4 Param Promoter (6B 配置)

```python
class ParamPromoter:
    """参数晋升模块 - 6B 配置"""
    
    DEFAULT_CONFIG = {
        'base_kl_weights': {
            'gap': 0.30,
            'policy': 0.30,
            'governance': 0.30,
            'writeback': 0.38,
        },
        'learning_rate': 1.0e-5,
        'replay_ratio': 0.35,
        'step1_max_change': 0.003,
        'step2_max_change': 0.008,
        'num_epochs': 3,
    }
    
    def promote(
        self,
        candidate: Candidate,
        config: Dict = None
    ) -> PromotionResult:
        """
        执行参数晋升
        
        Returns:
            PromotionResult: {
                'success': bool,
                'target_improvement': float,
                'old_ability_drop': float,
                'writeback_change': float,
                'steps': List[StepResult],
            }
        """
        pass
```

### 2.5 KB Promoter (整合)

```python
class KBPromoter:
    """知识库晋升模块"""
    
    def promote(self, candidate: Candidate) -> KBPromotionResult:
        """
        将候选存入知识库
        
        Returns:
            KBPromotionResult: {
                'success': bool,
                'kb_id': str,
                'retrievable': bool,
            }
        """
        pass
```

### 2.6 Rollback Hook (整合)

```python
class RollbackHook:
    """回滚钩子"""
    
    def save_baseline(self, model_state: Dict):
        """保存晋升前基线"""
        pass
    
    def check_and_rollback(
        self,
        current_metrics: Dict,
        threshold: float = 0.10
    ) -> bool:
        """
        检查是否需要回滚
        
        Returns:
            bool: 是否执行了回滚
        """
        pass
    
    def rollback(self) -> bool:
        """执行回滚"""
        pass
```

### 2.7 Metrics Collector (新建)

```python
class MetricsCollector:
    """统一指标收集器"""
    
    def collect_step(
        self,
        step_num: int,
        metrics: Dict
    ) -> None:
        """收集单步指标"""
        pass
    
    def get_cumulative_metrics(self) -> Dict:
        """获取累积指标"""
        pass
    
    def export_report(self) -> str:
        """导出完整报告"""
        pass
```

---

## 3. 配置管理

### 3.1 6B 默认配置

```python
# stage6_config.py

STAGE6_DEFAULT_CONFIG = {
    # 主干配置
    'backbone': {
        'model_path': '../phase19_native_backbone/checkpoints/native_128.pt',
        'device': 'cpu',
    },
    
    # 候选生成配置
    'candidate_generation': {
        'num_candidates': 5,
        'min_quality_score': 0.70,
    },
    
    # 类型分流配置
    'type_routing': {
        'param_promotion_threshold': 0.90,
        'kb_promotion_threshold': 0.80,
        'long_term_threshold': 0.70,
    },
    
    # 参数晋升配置 (6B)
    'param_promotion': {
        'base_kl_weights': {
            'gap': 0.30,
            'policy': 0.30,
            'governance': 0.30,
            'writeback': 0.38,
        },
        'learning_rate': 1.0e-5,
        'replay_ratio': 0.35,
        'step1_max_change': 0.003,
        'step2_max_change': 0.008,
        'num_epochs': 3,
        'kl_threshold_high': 0.05,
        'kl_threshold_low': 0.04,
    },
    
    # 回滚配置
    'rollback': {
        'old_ability_threshold': 0.10,
        'writeback_threshold': 0.05,
        'auto_rollback': True,
    },
    
    # 指标收集配置
    'metrics': {
        'collect_per_step': True,
        'export_format': 'json',
    },
}
```

### 3.2 环境变量覆盖

```python
# 允许通过环境变量覆盖配置
import os

def load_config():
    config = STAGE6_DEFAULT_CONFIG.copy()
    
    # 覆盖学习率
    if 'STAGE6_LR' in os.environ:
        config['param_promotion']['learning_rate'] = float(os.environ['STAGE6_LR'])
    
    # 覆盖 KL 权重
    if 'STAGE6_KL_GAP' in os.environ:
        config['param_promotion']['base_kl_weights']['gap'] = float(os.environ['STAGE6_KL_GAP'])
    
    return config
```

---

## 4. 错误处理与恢复

### 4.1 错误类型

```python
class Stage6Error(Exception):
    """Stage 6 基础错误"""
    pass

class BackboneError(Stage6Error):
    """主干推理错误"""
    pass

class CandidateGenerationError(Stage6Error):
    """候选生成错误"""
    pass

class PromotionError(Stage6Error):
    """晋升错误"""
    pass

class RollbackError(Stage6Error):
    """回滚错误"""
    pass
```

### 4.2 恢复策略

```python
RECOVERY_STRATEGIES = {
    'BackboneError': 'SKIP_AND_LOG',
    'CandidateGenerationError': 'SKIP_AND_LOG',
    'PromotionError': 'ROLLBACK_AND_RETRY',
    'RollbackError': 'ALERT_AND_HALT',
}
```

---

## 5. 验收标准

### 5.1 第一阶段验收（全链路整合）

| 验收项 | 标准 | 验证方法 |
|--------|------|----------|
| 6B 配置可运行 | 使用默认配置成功执行 | smoke_test.py |
| 1 步晋升 | 单步晋升成功，指标完整 | 运行 1 步测试 |
| 2 步晋升 | 连续两步晋升成功 | 运行 2 步测试 |
| 每步有指标 | 目标提升、旧能力掉落、writeback | 检查输出 |
| rollback 可用 | 回滚功能正常 | 触发回滚测试 |
| 无行为回退 | 端到端行为正常 | 对比测试 |

### 5.2 指标完整性检查

```python
REQUIRED_METRICS = [
    'step_number',
    'target_improvement',
    'old_ability_drop',
    'writeback_change',
    'retrieval_change',
    'policy_change',
    'governance_change',
    'kl_weights',
    'rollback_triggered',
    'timestamp',
]
```

---

## 6. 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-04-18 | 初始版本，基于 6B 配置 |

---

## 7. 参考文档

- [STAGE6_ENTRY.md](STAGE6_ENTRY.md) - Stage 6 入口文档
- [eval/stage5g_final_report.md](eval/stage5g_final_report.md) - Stage 5G 最终报告
- [stage5g_exp6b_stronger_protection.py](stage5g_exp6b_stronger_protection.py) - 6B 配置实现
