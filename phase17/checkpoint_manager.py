"""
Checkpoint Manager v1 - 检查点管理器 v1

Phase 17 Stage 2: 本地真实训练基础设施
目标：保存/恢复训练状态，支持 latest/best/experimental 区分
"""

import torch
import json
import shutil
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class CheckpointMetadata:
    """检查点元数据"""
    checkpoint_id: str
    checkpoint_type: str  # 'latest', 'best', 'experimental'
    epoch: int
    step: int
    timestamp: str
    metrics: Dict[str, float]
    config: Dict[str, Any]


class CheckpointManager:
    """检查点管理器"""
    
    def __init__(
        self,
        checkpoint_dir: str = "phase17/checkpoints",
        keep_last_n: int = 3,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        
        # 子目录
        self.latest_dir = self.checkpoint_dir / "latest"
        self.best_dir = self.checkpoint_dir / "best"
        self.experimental_dir = self.checkpoint_dir / "experimental"
        
        for dir_path in [self.latest_dir, self.best_dir, self.experimental_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # 跟踪最佳指标
        self.best_metric_value = float('-inf')
        self.best_checkpoint_path = None
    
    def save_checkpoint(
        self,
        model_state: Dict[str, torch.Tensor],
        optimizer_state: Dict[str, Any],
        epoch: int,
        step: int,
        metrics: Dict[str, float],
        config: Dict[str, Any],
        checkpoint_type: str = "experimental",
    ) -> str:
        """
        保存检查点
        
        Args:
            model_state: 模型状态字典
            optimizer_state: 优化器状态字典
            epoch: 当前轮次
            step: 当前步数
            metrics: 当前指标
            config: 训练配置
            checkpoint_type: 检查点类型 ('latest', 'best', 'experimental')
        
        Returns:
            checkpoint_path: 保存的检查点路径
        """
        # 生成检查点ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_id = f"checkpoint_{checkpoint_type}_e{epoch}_s{step}_{timestamp}"
        
        # 确定保存目录
        if checkpoint_type == "latest":
            save_dir = self.latest_dir
        elif checkpoint_type == "best":
            save_dir = self.best_dir
        else:
            save_dir = self.experimental_dir
        
        checkpoint_path = save_dir / f"{checkpoint_id}.pt"
        
        # 准备元数据
        metadata = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            checkpoint_type=checkpoint_type,
            epoch=epoch,
            step=step,
            timestamp=timestamp,
            metrics=metrics,
            config=config,
        )
        
        # 保存检查点
        checkpoint = {
            'model_state': model_state,
            'optimizer_state': optimizer_state,
            'epoch': epoch,
            'step': step,
            'metrics': metrics,
            'metadata': asdict(metadata),
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        # 同时保存元数据为JSON（方便查看）
        metadata_path = save_dir / f"{checkpoint_id}.json"
        with open(metadata_path, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)
        
        # 如果是latest，清理旧的latest检查点
        if checkpoint_type == "latest":
            self._cleanup_old_checkpoints(self.latest_dir, keep_last=1)
        
        # 如果是experimental，清理旧的experimental检查点
        if checkpoint_type == "experimental":
            self._cleanup_old_checkpoints(self.experimental_dir, keep_last=self.keep_last_n)
        
        print(f"✓ 检查点已保存: {checkpoint_path}")
        return str(checkpoint_path)
    
    def save_best_checkpoint(
        self,
        model_state: Dict[str, torch.Tensor],
        optimizer_state: Dict[str, Any],
        epoch: int,
        step: int,
        metric_value: float,
        metric_name: str = "overall_score",
        config: Dict[str, Any] = None,
    ) -> Optional[str]:
        """
        根据指标保存最佳检查点
        
        Returns:
            checkpoint_path: 如果保存了新的最佳检查点，返回路径；否则返回None
        """
        if metric_value <= self.best_metric_value:
            return None
        
        self.best_metric_value = metric_value
        
        metrics = {metric_name: metric_value}
        
        checkpoint_path = self.save_checkpoint(
            model_state=model_state,
            optimizer_state=optimizer_state,
            epoch=epoch,
            step=step,
            metrics=metrics,
            config=config or {},
            checkpoint_type="best",
        )
        
        self.best_checkpoint_path = checkpoint_path
        print(f"🎉 新的最佳检查点! {metric_name}={metric_value:.4f}")
        
        return checkpoint_path
    
    def load_checkpoint(
        self,
        checkpoint_path: Optional[str] = None,
        checkpoint_type: str = "latest",
    ) -> Dict[str, Any]:
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点路径，如果为None则根据checkpoint_type自动选择
            checkpoint_type: 检查点类型 ('latest', 'best', 'experimental')
        
        Returns:
            checkpoint: 检查点字典
        """
        if checkpoint_path is None:
            checkpoint_path = self._get_latest_checkpoint_path(checkpoint_type)
        
        if checkpoint_path is None:
            raise FileNotFoundError(f"未找到类型为 '{checkpoint_type}' 的检查点")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print(f"✓ 检查点已加载: {checkpoint_path}")
        print(f"  Epoch: {checkpoint['epoch']}, Step: {checkpoint['step']}")
        print(f"  Metrics: {checkpoint['metrics']}")
        
        return checkpoint
    
    def _get_latest_checkpoint_path(self, checkpoint_type: str) -> Optional[str]:
        """获取最新的检查点路径"""
        if checkpoint_type == "latest":
            search_dir = self.latest_dir
        elif checkpoint_type == "best":
            search_dir = self.best_dir
        else:
            search_dir = self.experimental_dir
        
        checkpoints = list(search_dir.glob("checkpoint_*.pt"))
        if not checkpoints:
            return None
        
        # 按修改时间排序，返回最新的
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        return str(latest)
    
    def _cleanup_old_checkpoints(self, directory: Path, keep_last: int):
        """清理旧的检查点"""
        checkpoints = list(directory.glob("checkpoint_*.pt"))
        
        if len(checkpoints) <= keep_last:
            return
        
        # 按修改时间排序
        checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        # 删除旧的
        for old_checkpoint in checkpoints[keep_last:]:
            old_checkpoint.unlink()
            # 同时删除对应的JSON文件
            json_file = old_checkpoint.with_suffix('.json')
            if json_file.exists():
                json_file.unlink()
            print(f"  清理旧检查点: {old_checkpoint.name}")
    
    def list_checkpoints(self, checkpoint_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出所有检查点"""
        checkpoints = []
        
        dirs = []
        if checkpoint_type is None or checkpoint_type == "latest":
            dirs.append(self.latest_dir)
        if checkpoint_type is None or checkpoint_type == "best":
            dirs.append(self.best_dir)
        if checkpoint_type is None or checkpoint_type == "experimental":
            dirs.append(self.experimental_dir)
        
        for dir_path in dirs:
            for json_file in dir_path.glob("checkpoint_*.json"):
                with open(json_file, 'r') as f:
                    metadata = json.load(f)
                    checkpoints.append(metadata)
        
        # 按时间排序
        checkpoints.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return checkpoints
    
    def get_best_checkpoint_info(self) -> Optional[Dict[str, Any]]:
        """获取最佳检查点信息"""
        if self.best_checkpoint_path is None:
            # 尝试从目录中加载
            best_checkpoints = list(self.best_dir.glob("checkpoint_*.json"))
            if not best_checkpoints:
                return None
            
            latest_best = max(best_checkpoints, key=lambda p: p.stat().st_mtime)
            with open(latest_best, 'r') as f:
                return json.load(f)
        
        # 从已保存的路径加载
        metadata_path = self.best_checkpoint_path.replace('.pt', '.json')
        with open(metadata_path, 'r') as f:
            return json.load(f)


# 便捷函数
def create_checkpoint_manager(
    checkpoint_dir: str = "phase17/checkpoints",
    keep_last_n: int = 3,
) -> CheckpointManager:
    """创建检查点管理器"""
    return CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        keep_last_n=keep_last_n,
    )


# 测试
if __name__ == "__main__":
    print("="*70)
    print("Checkpoint Manager v1 - 测试")
    print("="*70)
    
    manager = create_checkpoint_manager()
    
    # 模拟模型和优化器状态
    model_state = {
        'layer1.weight': torch.randn(10, 10),
        'layer1.bias': torch.randn(10),
    }
    
    optimizer_state = {
        'state': {},
        'param_groups': [{'lr': 0.001}],
    }
    
    # 保存检查点
    print("\n1. 保存实验检查点...")
    checkpoint_path = manager.save_checkpoint(
        model_state=model_state,
        optimizer_state=optimizer_state,
        epoch=1,
        step=100,
        metrics={'loss': 0.5, 'accuracy': 0.8},
        config={'lr': 0.001, 'batch_size': 32},
        checkpoint_type="experimental",
    )
    
    # 保存最佳检查点
    print("\n2. 保存最佳检查点...")
    best_path = manager.save_best_checkpoint(
        model_state=model_state,
        optimizer_state=optimizer_state,
        epoch=2,
        step=200,
        metric_value=0.85,
        metric_name="accuracy",
    )
    
    # 列出检查点
    print("\n3. 列出所有检查点...")
    checkpoints = manager.list_checkpoints()
    for cp in checkpoints:
        print(f"  - {cp['checkpoint_id']} (type={cp['checkpoint_type']}, epoch={cp['epoch']})")
    
    # 加载检查点
    print("\n4. 加载最新检查点...")
    loaded = manager.load_checkpoint(checkpoint_type="experimental")
    print(f"  加载的epoch: {loaded['epoch']}")
    print(f"  加载的metrics: {loaded['metrics']}")
    
    print("\n✓ 检查点管理器工作正常")
