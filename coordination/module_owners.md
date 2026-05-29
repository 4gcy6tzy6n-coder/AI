# 模块负责人

## 概述

本文档定义各模块的负责人，明确职责边界和协作方式。

## 模块划分

### 核心引擎模块

| 模块 | 负责人 | 协作 AI | 状态 |
|------|--------|---------|------|
| Unit 系统 | TBD | Backend AI | 规划中 |
| 思考引擎 | TBD | Backend AI | 规划中 |
| 检索引擎 | TBD | Retrieval AI | 规划中 |
| 记忆调度器 | TBD | Backend AI | 规划中 |
| TSLA 引擎 | TBD | TSLA AI | 规划中 |
| 门控系统 | TBD | Backend AI | 规划中 |

### 支持模块

| 模块 | 负责人 | 协作 AI | 状态 |
|------|--------|---------|------|
| 训练系统 | TBD | Experiment AI | 规划中 |
| 存储调度器 | TBD | Backend AI | 规划中 |
| 贝叶斯系统 | TBD | Backend AI | 规划中 |

### 接口模块

| 模块 | 负责人 | 协作 AI | 状态 |
|------|--------|---------|------|
| CLI | TBD | Backend AI | 规划中 |
| API | TBD | Backend AI | 规划中 |
| UI 后端 | TBD | Backend AI | 规划中 |

## 协作规范

### 工作交接流程

1. **任务分配**: 在 coordination/handoff/incoming/ 创建任务文件
2. **工作执行**: 负责人完成任务
3. **结果提交**: 将结果放入 coordination/handoff/archived/
4. **状态更新**: 更新本文档和 milestone_status.md

### 沟通渠道

- **日常沟通**: 通过 sync_logs/ 记录
- **问题升级**: 创建 handoff 文件
- **决策记录**: 更新 ADR 文档

### 代码审查

- 所有代码变更需通过 review_checklists/code_review.md
- 规范变更需通过 review_checklists/spec_review.md
- 实验结果需通过 review_checklists/experiment_review.md

## 当前活跃任务

暂无

## 历史任务

暂无
