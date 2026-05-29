# Post Transformer AI

## 项目概述

Post Transformer AI 是一个下一代人工智能系统架构，旨在突破传统 Transformer 模型的限制，实现更高效、更可靠的推理和记忆机制。

## 项目结构

本项目严格遵循预定义的目录结构和文件组织规范。所有代码、文档和配置文件必须放置在指定的位置。

```
post_transformer_ai/
├── README.md                 # 项目说明
├── LICENSE                   # 许可证
├── .gitignore               # Git 忽略规则
├── pyproject.toml           # Python 项目配置
├── requirements.txt         # 依赖列表
├── Makefile                 # 构建脚本
│
├── docs/                    # 正式文档
├── coordination/            # 多 AI 协同区
├── data/                    # 数据目录
├── configs/                 # 配置文件
├── src/                     # 源代码
├── tests/                   # 测试代码
├── experiments/             # 实验目录
├── scripts/                 # 脚本文件
├── reports/                 # 报告目录
└── artifacts/               # 构建产物
```

## 开发规范

### 文件放置规则

1. **严格遵循目录结构**：所有文件必须放置在指定的目录中
2. **文档分离**：正式文档放入 `docs/`，临时说明放入 `coordination/`
3. **代码组织**：源代码按功能模块组织在 `src/core/` 下
4. **测试分类**：测试按类型分类在 `tests/` 下

### 命名规范

- 文件使用小写字母和下划线
- Python 模块使用 `snake_case`
- 文档使用 `kebab-case`
- 实验目录使用 `exp_XXX_描述` 格式

### 开发流程

1. 查阅 `docs/` 了解系统设计和规范
2. 查看 `coordination/module_owners.md` 确认模块负责人
3. 遵循 `coordination/prompts/` 中的 AI 提示词规范
4. 提交前使用 `coordination/review_checklists/` 进行自查

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行测试
make test

# 运行实验
make run-exp EXP=exp_001_unit_pipeline
```

## 许可证

[LICENSE](LICENSE)
