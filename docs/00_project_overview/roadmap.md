# 项目路线图

## Current Roadmap Addendum: Stage22-30

The early M1-M12 roadmap below is retained for historical context. The active
project roadmap has moved to the governed offline evolution baseline chain.
The current long-horizon target is Stage30: a governed production baseline
freeze after Stage21-B through Stage29 have completed and frozen their
validation evidence.

Current planning documents:

- `docs/00_project_overview/stage22_to_stage30_roadmap.md`
- `docs/05_decisions/stage30_completion_definition.md`
- `docs/05_decisions/stage30_preparation_plan.md`

Active sequence:

```text
Stage21-B -> Stage22 -> Stage23 -> Stage24 -> Stage25 -> Stage26 -> Stage27 -> Stage28 -> Stage29 -> Stage30
```

## Phase 1: 基础架构 (M1-M3)

### M1: 核心概念验证
- [ ] Unit 定义和解析器实现
- [ ] 基础思考引擎原型
- [ ] 简单记忆存储实现

### M2: 引擎开发
- [ ] 思考引擎 v1 完成
- [ ] 检索引擎 v1 完成
- [ ] 记忆调度器 v1 完成

### M3: 系统集成
- [ ] 引擎间通信协议
- [ ] 基础流水线实现
- [ ] 初步评估框架

## Phase 2: 功能完善 (M4-M6)

### M4: TSLA 系统
- [ ] TSLA 评分器实现
- [ ] 硬否决机制
- [ ] 动作路由器

### M5: 门控机制
- [ ] 写入门实现
- [ ] 稳定性门实现
- [ ] 提升门实现

### M6: 训练系统
- [ ] 训练案例构建器
- [ ] 噪声注入器
- [ ] 自举训练流程

## Phase 3: 优化与扩展 (M7-M9)

### M7: 存储优化
- [ ] GPU 缓存管理
- [ ] RAM 缓存优化
- [ ] SSD 索引系统

### M8: 贝叶斯更新
- [ ] 先验管理系统
- [ ] 后验追踪
- [ ] 不确定性量化

### M9: 实验与评估
- [ ] 完整评估流水线
- [ ] 消融实验
- [ ] 复杂度评估

## Phase 4: 生产准备 (M10-M12)

### M10: 接口完善
- [ ] CLI 工具
- [ ] API 服务
- [ ] UI 后端

### M11: 性能优化
- [ ] 延迟优化
- [ ] 吞吐量提升
- [ ] 资源效率

### M12: 部署准备
- [ ] 文档完善
- [ ] 监控仪表板
- [ ] 生产部署指南
