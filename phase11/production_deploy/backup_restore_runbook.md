# Backup & Restore Runbook - 备份恢复手册

**版本**: v1.0  
**日期**: 2026-04-18  
**阶段**: Phase 11 - WP2: 生产部署基线建立

---

## 1. 概述

本文档定义系统的备份恢复流程，包括配置备份、数据备份、索引备份和灾难恢复。

### 1.1 备份策略

| 类型 | 频率 | 保留期 | 存储位置 |
|------|------|--------|----------|
| 配置备份 | 每 6 小时 | 30 天 | Git + S3 |
| 数据备份 | 每天 2 AM | 7 天 | S3 |
| 索引备份 | 每天 3 AM | 3 天 | S3 |
| 快照 | 每周日 4 AM | 4 周 | S3 |

### 1.2 RTO/RPO 目标

| 场景 | RTO | RPO |
|------|-----|-----|
| 单服务故障 | 5 分钟 | 0 |
| 单节点故障 | 10 分钟 | 0 |
| 数据损坏 | 1 小时 | 1 小时 |
| 全集群故障 | 4 小时 | 1 小时 |

---

## 2. 配置备份

### 2.1 自动备份

```bash
# 配置备份脚本
./scripts/backup-config.sh
```

备份内容：
- Kubernetes manifests
- ConfigMaps
- Secrets (加密)
- 环境变量配置

### 2.2 手动备份

```bash
# 导出所有配置
kubectl get all -n post-transformer-ai -o yaml > backup/all-resources.yaml

# 导出 ConfigMaps
kubectl get configmaps -n post-transformer-ai -o yaml > backup/configmaps.yaml

# 导出 Secrets (加密)
kubectl get secrets -n post-transformer-ai -o yaml > backup/secrets.yaml
```

### 2.3 配置恢复

```bash
# 恢复配置
kubectl apply -f backup/all-resources.yaml

# 验证恢复
kubectl get all -n post-transformer-ai
```

---

## 3. 数据备份

### 3.1 PostgreSQL 备份

#### 自动备份

```bash
# 创建备份
pg_dump -h postgresql -U postgres memory_service > backup/memory_service_$(date +%Y%m%d).sql

# 压缩备份
gzip backup/memory_service_$(date +%Y%m%d).sql

# 上传到 S3
aws s3 cp backup/memory_service_$(date +%Y%m%d).sql.gz s3://pta-backups/data/
```

#### 手动备份

```bash
# 完整备份
kubectl exec -it postgresql-0 -- pg_dumpall -U postgres > backup/full_backup.sql

# 单库备份
kubectl exec -it postgresql-0 -- pg_dump -U postgres memory_service > backup/memory_service.sql
```

#### 数据恢复

```bash
# 从 S3 下载
aws s3 cp s3://pta-backups/data/memory_service_20260418.sql.gz backup/
gunzip backup/memory_service_20260418.sql.gz

# 恢复数据
kubectl exec -i postgresql-0 -- psql -U postgres memory_service < backup/memory_service_20260418.sql

# 验证恢复
kubectl exec -it postgresql-0 -- psql -U postgres -c "SELECT COUNT(*) FROM memory_objects;"
```

### 3.2 Redis 备份

#### 备份

```bash
# 执行 BGSAVE
kubectl exec -it redis-0 -- redis-cli BGSAVE

# 复制 RDB 文件
kubectl cp redis-0:/data/dump.rdb backup/redis_dump.rdb

# 上传到 S3
aws s3 cp backup/redis_dump.rdb s3://pta-backups/redis/
```

#### 恢复

```bash
# 从 S3 下载
aws s3 cp s3://pta-backups/redis/redis_dump_20260418.rdb backup/

# 停止 Redis
kubectl scale statefulset redis --replicas=0

# 复制 RDB 文件到 Pod
kubectl cp backup/redis_dump_20260418.rdb redis-0:/data/dump.rdb

# 启动 Redis
kubectl scale statefulset redis --replicas=3
```

---

## 4. 索引备份

### 4.1 Elasticsearch 快照

#### 创建快照仓库

```bash
# 注册快照仓库
curl -X PUT "localhost:9200/_snapshot/pta_backup" -H 'Content-Type: application/json' -d'
{
  "type": "s3",
  "settings": {
    "bucket": "pta-backups",
    "base_path": "elasticsearch",
    "region": "us-east-1"
  }
}'
```

#### 创建快照

```bash
# 手动创建快照
curl -X PUT "localhost:9200/_snapshot/pta_backup/snapshot_$(date +%Y%m%d)"

# 查看快照状态
curl -X GET "localhost:9200/_snapshot/pta_backup/snapshot_20260418"
```

#### 恢复快照

```bash
# 关闭索引
curl -X POST "localhost:9200/memory_index/_close"

# 恢复快照
curl -X POST "localhost:9200/_snapshot/pta_backup/snapshot_20260418/_restore"

# 打开索引
curl -X POST "localhost:9200/memory_index/_open"

# 验证恢复
curl -X GET "localhost:9200/memory_index/_count"
```

---

## 5. 灾难恢复

### 5.1 灾难恢复流程

```
1. 评估影响范围
   - 确定故障类型
   - 评估数据丢失范围
   - 确定恢复优先级

2. 启动灾难恢复
   - 通知相关团队
   - 启动 DR 流程
   - 分配恢复任务

3. 基础设施恢复
   - 恢复 Kubernetes 集群
   - 恢复网络配置
   - 恢复存储

4. 数据恢复
   - 恢复 PostgreSQL
   - 恢复 Redis
   - 恢复 Elasticsearch

5. 应用恢复
   - 部署应用服务
   - 验证配置
   - 健康检查

6. 验证恢复
   - 功能测试
   - 数据完整性检查
   - 性能测试

7. 恢复服务
   - 切换流量
   - 监控运行状态
   - 通知用户
```

### 5.2 全集群恢复

```bash
#!/bin/bash
# disaster-recovery.sh

set -e

echo "Starting disaster recovery..."

# 1. 恢复 Kubernetes 集群
echo "Restoring Kubernetes cluster..."
# 根据集群类型执行恢复

# 2. 恢复命名空间
echo "Restoring namespace..."
kubectl create namespace post-transformer-ai

# 3. 恢复配置
echo "Restoring configurations..."
kubectl apply -f backup/configmaps.yaml
kubectl apply -f backup/secrets.yaml

# 4. 恢复基础设施
echo "Restoring infrastructure..."
kubectl apply -f manifests/infrastructure/

# 等待基础设施就绪
echo "Waiting for infrastructure..."
kubectl wait --for=condition=ready pod -l app=postgresql --timeout=300s
kubectl wait --for=condition=ready pod -l app=redis --timeout=300s
kubectl wait --for=condition=ready pod -l app=elasticsearch --timeout=300s

# 5. 恢复数据
echo "Restoring data..."
./scripts/restore-data.sh

# 6. 部署应用
echo "Deploying applications..."
kubectl apply -f manifests/applications/

# 等待应用就绪
echo "Waiting for applications..."
kubectl wait --for=condition=ready pod -l app=api-gateway --timeout=300s

# 7. 验证恢复
echo "Verifying recovery..."
./scripts/verify-recovery.sh

echo "Disaster recovery completed!"
```

### 5.3 数据恢复脚本

```bash
#!/bin/bash
# restore-data.sh

echo "Restoring data..."

# 恢复 PostgreSQL
echo "Restoring PostgreSQL..."
LATEST_BACKUP=$(aws s3 ls s3://pta-backups/data/ | sort | tail -1 | awk '{print $4}')
aws s3 cp s3://pta-backups/data/$LATEST_BACKUP backup/
gunzip backup/$LATEST_BACKUP

kubectl exec -i postgresql-0 -- psql -U postgres < backup/${LATEST_BACKUP%.gz}

# 恢复 Redis
echo "Restoring Redis..."
LATEST_REDIS=$(aws s3 ls s3://pta-backups/redis/ | sort | tail -1 | awk '{print $4}')
aws s3 cp s3://pta-backups/redis/$LATEST_REDIS backup/

kubectl scale statefulset redis --replicas=0
kubectl cp backup/$LATEST_REDIS redis-0:/data/dump.rdb
kubectl scale statefulset redis --replicas=3

# 恢复 Elasticsearch
echo "Restoring Elasticsearch..."
LATEST_SNAPSHOT=$(curl -s localhost:9200/_snapshot/pta_backup/_all | jq -r '.snapshots[-1].snapshot')
curl -X POST "localhost:9200/_snapshot/pta_backup/$LATEST_SNAPSHOT/_restore"

echo "Data restoration completed!"
```

---

## 6. 验证恢复

### 6.1 功能验证

```bash
#!/bin/bash
# verify-recovery.sh

echo "Verifying recovery..."

# 1. 检查 Pod 状态
echo "Checking pod status..."
kubectl get pods -n post-transformer-ai

# 2. 检查服务状态
echo "Checking service health..."
curl -f http://api-gateway:8080/health
curl -f http://retrieval-service:8081/health
curl -f http://governance-service:8082/health
curl -f http://memory-service:8083/health

# 3. 检查数据库连接
echo "Checking database connectivity..."
kubectl exec -it postgresql-0 -- psql -U postgres -c "SELECT 1;"

# 4. 检查数据完整性
echo "Checking data integrity..."
kubectl exec -it postgresql-0 -- psql -U postgres -c "SELECT COUNT(*) FROM memory_objects;"

# 5. 端到端测试
echo "Running end-to-end test..."
./scripts/e2e-test.sh

echo "Verification completed!"
```

### 6.2 数据完整性检查

```sql
-- 检查记忆对象数量
SELECT layer, COUNT(*) as count 
FROM memory_objects 
GROUP BY layer;

-- 检查事务记录
SELECT status, COUNT(*) as count 
FROM transactions 
WHERE created_at > NOW() - INTERVAL '24 hours'
GROUP BY status;

-- 检查治理事件
SELECT action, COUNT(*) as count 
FROM governance_events 
WHERE created_at > NOW() - INTERVAL '24 hours'
GROUP BY action;
```

---

## 7. 应急联系

| 角色 | 联系人 | 联系方式 |
|------|--------|----------|
| 运维负责人 | | |
| 开发负责人 | | |
| DBA | | |
| 安全负责人 | | |

---

## 8. 版本历史

| 版本 | 日期 | 修改内容 |
|------|------|----------|
| v1.0 | 2026-04-18 | 初始版本，定义备份恢复流程 |

---

**文档状态**: 生效中  
**定期演练，确保恢复流程有效**  
**负责人**: Phase 11 WP2 负责人
