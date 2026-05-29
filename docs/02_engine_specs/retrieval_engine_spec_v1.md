# 检索引擎规范 v1

## 概述

检索引擎负责从内部记忆和外部知识源获取相关信息，为思考引擎提供证据支持。

## 职责

1. 内部记忆检索
2. 外部知识检索
3. 结果排序和过滤
4. 证据包构建
5. 相关性评分

## 架构

```
┌─────────────────────────────────────┐
│       Retrieval Engine              │
├─────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  │
│  │  Internal   │  │  External   │  │
│  │  Retriever  │  │  Retriever  │  │
│  └──────┬──────┘  └──────┬──────┘  │
│         │                │         │
│         └────────┬───────┘         │
│                  ▼                 │
│         ┌─────────────┐            │
│         │   Ranking   │            │
│         └──────┬──────┘            │
│                │                   │
│                ▼                   │
│         ┌─────────────┐            │
│         │Evidence Pack│            │
│         │  Builder    │            │
│         └─────────────┘            │
└─────────────────────────────────────┘
```

## 组件规范

### 1. Internal Retriever (内部检索器)

**输入**: Query (查询)
**输出**: list[InternalResult] (内部结果列表)

**功能**:
- 瞬态记忆查询
- 长期记忆查询
- 永久记忆查询
- 索引管理

**接口**:
```python
class InternalRetriever:
    def retrieve(
        self, 
        query: str, 
        zones: list[MemoryZone],
        top_k: int = 10
    ) -> list[InternalResult]:
        """从内部记忆检索"""
        pass
    
    def update_index(self, memory: Memory):
        """更新索引"""
        pass
```

### 2. External Retriever (外部检索器)

**输入**: Query (查询)
**输出**: list[ExternalResult] (外部结果列表)

**功能**:
- 知识库查询
- API 调用
- 文档检索
- 缓存管理

**接口**:
```python
class ExternalRetriever:
    def retrieve(
        self,
        query: str,
        sources: list[Source],
        timeout: int = 5000
    ) -> list[ExternalResult]:
        """从外部源检索"""
        pass
    
    def check_cache(self, query: str) -> Optional[list[ExternalResult]]:
        """检查缓存"""
        pass
```

### 3. Ranking (排序器)

**输入**: list[Result] (混合结果)
**输出**: list[RankedResult] (排序后结果)

**功能**:
- 相关性评分
- 来源可信度评估
- 时效性评估
- 去重和合并

**接口**:
```python
class Ranking:
    def rank(
        self,
        results: list[Result],
        query: str,
        strategy: RankingStrategy = "hybrid"
    ) -> list[RankedResult]:
        """排序结果"""
        pass
    
    def calculate_relevance(
        self,
        result: Result,
        query: str
    ) -> float:
        """计算相关性"""
        pass
```

### 4. Evidence Pack Builder (证据包构建器)

**输入**: list[RankedResult] (排序后结果)
**输出**: EvidencePack (证据包)

**功能**:
- 证据筛选
- 证据组织
- 冲突标记
- 引用生成

**接口**:
```python
class EvidencePackBuilder:
    def build(
        self,
        results: list[RankedResult],
        max_evidence: int = 10
    ) -> EvidencePack:
        """构建证据包"""
        pass
    
    def detect_conflicts(
        self,
        evidence: list[Evidence]
    ) -> list[Conflict]:
        """检测证据冲突"""
        pass
```

## 数据模型

### InternalResult
```python
class InternalResult:
    result_id: str
    source_zone: MemoryZone
    memory_id: str
    content: str
    relevance_score: float
    access_count: int
    last_accessed: datetime
```

### ExternalResult
```python
class ExternalResult:
    result_id: str
    source: Source
    content: str
    url: Optional[str]
    relevance_score: float
    freshness_score: float
    source_credibility: float
```

### EvidencePack
```python
class EvidencePack:
    pack_id: str
    query: str
    evidence: list[Evidence]
    conflicts: list[Conflict]
    coverage_score: float
    overall_confidence: float
```

### Evidence
```python
class Evidence:
    evidence_id: str
    content: str
    source: str
    relevance: float
    credibility: float
    timestamp: datetime
    citations: list[str]
```

## 配置参数

```yaml
retrieval_engine:
  internal:
    enabled: true
    zones: ["transient", "long_term", "shallow_permanent"]
    top_k: 20
    min_relevance: 0.3
  
  external:
    enabled: true
    sources: ["knowledge_base", "api", "documents"]
    timeout_ms: 5000
    cache_ttl: 3600
    max_results_per_source: 10
  
  ranking:
    strategy: "hybrid"  # semantic, lexical, hybrid
    relevance_weight: 0.5
    credibility_weight: 0.3
    freshness_weight: 0.2
  
  evidence_pack:
    max_evidence: 10
    min_evidence_relevance: 0.5
    include_conflicts: true
```

## 性能要求

- 内部检索延迟: < 100ms
- 外部检索延迟: < 1000ms
- 排序延迟: < 50ms
- 并发检索数: 支持 20 个并发
