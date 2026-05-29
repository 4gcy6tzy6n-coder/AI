"""
Stage 6 Full Rollback Snapshot

Rollback 完整快照 - 保存和恢复完整的系统状态

包含:
1. Model parameters
2. Optimizer state
3. Random number generator state
4. 运行时缓存和状态
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import random
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import copy
import pickle

from phase19_native_backbone.native_backbone_tiny_v1 import NativeBackboneTinyV1


@dataclass
class FullSystemSnapshot:
    """完整系统快照"""
    # 元数据 (带默认值的放前面)
    timestamp: str = ""
    version: str = "1.0"
    
    # 模型状态
    model_state: Dict[str, Any] = None
    
    # 优化器状态
    optimizer_state: Optional[Dict[str, Any]] = None
    
    # 随机数状态
    python_rng_state: Any = None
    numpy_rng_state: Any = None
    torch_rng_state: torch.Tensor = None
    torch_cuda_rng_state: Optional[Any] = None
    
    # 运行时状态
    runtime_state: Dict[str, Any] = None
    
    # 元数据
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict:
        """转换为字典 (用于序列化)"""
        return {
            'timestamp': self.timestamp,
            'version': self.version,
            'model_state': self.model_state,
            'optimizer_state': self.optimizer_state,
            'python_rng_state': self.python_rng_state,
            'numpy_rng_state': self.numpy_rng_state,
            'torch_rng_state': self.torch_rng_state.tolist() if isinstance(self.torch_rng_state, torch.Tensor) else self.torch_rng_state,
            'torch_cuda_rng_state': self.torch_cuda_rng_state,
            'runtime_state': self.runtime_state,
            'metadata': self.metadata,
        }


class FullRollbackManager:
    """
    完整回滚管理器
    
    保存和恢复完整的系统状态，确保完全可复现
    """
    
    def __init__(self, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None):
        self.model = model
        self.optimizer = optimizer
        self.snapshots: Dict[str, FullSystemSnapshot] = {}
        self.current_snapshot_id: Optional[str] = None
    
    def create_snapshot(self, name: str, metadata: Dict = None) -> str:
        """
        创建完整快照
        
        Args:
            name: 快照名称
            metadata: 额外元数据
            
        Returns:
            快照ID
        """
        snapshot_id = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"\n创建快照: {snapshot_id}")
        print("-" * 50)
        
        # 1. 保存模型状态
        print("[1/5] 保存模型状态...")
        model_state = copy.deepcopy(self.model.state_dict())
        
        # 2. 保存优化器状态
        print("[2/5] 保存优化器状态...")
        optimizer_state = None
        if self.optimizer is not None:
            optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        
        # 3. 保存随机数状态
        print("[3/5] 保存随机数状态...")
        python_rng_state = random.getstate()
        numpy_rng_state = np.random.get_state()
        torch_rng_state = torch.get_rng_state()
        
        torch_cuda_rng_state = None
        if torch.cuda.is_available():
            torch_cuda_rng_state = torch.cuda.get_rng_state_all()
        
        # 4. 保存运行时状态
        print("[4/5] 保存运行时状态...")
        runtime_state = self._capture_runtime_state()
        
        # 5. 构建快照
        print("[5/5] 构建快照...")
        snapshot = FullSystemSnapshot(
            timestamp=datetime.now().isoformat(),
            model_state=model_state,
            optimizer_state=optimizer_state,
            python_rng_state=python_rng_state,
            numpy_rng_state=numpy_rng_state,
            torch_rng_state=torch_rng_state,
            torch_cuda_rng_state=torch_cuda_rng_state,
            runtime_state=runtime_state,
            metadata=metadata or {},
        )
        
        self.snapshots[snapshot_id] = snapshot
        self.current_snapshot_id = snapshot_id
        
        print(f"✓ 快照创建完成: {snapshot_id}")
        
        return snapshot_id
    
    def restore_snapshot(self, snapshot_id: str) -> bool:
        """
        恢复快照
        
        Args:
            snapshot_id: 要恢复的快照ID
            
        Returns:
            是否成功
        """
        if snapshot_id not in self.snapshots:
            print(f"错误: 快照 {snapshot_id} 不存在")
            return False
        
        snapshot = self.snapshots[snapshot_id]
        
        print(f"\n恢复快照: {snapshot_id}")
        print("-" * 50)
        
        try:
            # 1. 恢复模型状态
            print("[1/5] 恢复模型状态...")
            self.model.load_state_dict(snapshot.model_state)
            
            # 2. 恢复优化器状态
            print("[2/5] 恢复优化器状态...")
            if self.optimizer is not None and snapshot.optimizer_state is not None:
                self.optimizer.load_state_dict(snapshot.optimizer_state)
            
            # 3. 恢复随机数状态
            print("[3/5] 恢复随机数状态...")
            random.setstate(snapshot.python_rng_state)
            np.random.set_state(snapshot.numpy_rng_state)
            torch.set_rng_state(snapshot.torch_rng_state)
            
            if torch.cuda.is_available() and snapshot.torch_cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(snapshot.torch_cuda_rng_state)
            
            # 4. 恢复运行时状态
            print("[4/5] 恢复运行时状态...")
            self._restore_runtime_state(snapshot.runtime_state)
            
            print("[5/5] 验证恢复...")
            # 验证模型状态
            current_state = self.model.state_dict()
            match = all(
                torch.allclose(current_state[k], snapshot.model_state[k])
                for k in snapshot.model_state.keys()
            )
            
            if match:
                print(f"✓ 快照恢复完成: {snapshot_id}")
                return True
            else:
                print(f"✗ 快照恢复验证失败")
                return False
                
        except Exception as e:
            print(f"✗ 恢复快照失败: {e}")
            return False
    
    def _capture_runtime_state(self) -> Dict[str, Any]:
        """捕获运行时状态"""
        state = {
            'model_training_mode': self.model.training,
            'timestamp': datetime.now().isoformat(),
        }
        
        # 捕获模型特定状态 (如 batch norm 运行统计)
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                if hasattr(module, 'running_mean'):
                    state[f'{name}_running_mean'] = module.running_mean.clone()
                if hasattr(module, 'running_var'):
                    state[f'{name}_running_var'] = module.running_var.clone()
                if hasattr(module, 'num_batches_tracked'):
                    state[f'{name}_num_batches_tracked'] = module.num_batches_tracked.clone()
        
        return state
    
    def _restore_runtime_state(self, state: Dict[str, Any]):
        """恢复运行时状态"""
        # 恢复训练模式
        if state.get('model_training_mode', False):
            self.model.train()
        else:
            self.model.eval()
        
        # 恢复 batch norm 统计
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                if f'{name}_running_mean' in state:
                    module.running_mean = state[f'{name}_running_mean']
                if f'{name}_running_var' in state:
                    module.running_var = state[f'{name}_running_var']
                if f'{name}_num_batches_tracked' in state:
                    module.num_batches_tracked = state[f'{name}_num_batches_tracked']
    
    def save_to_file(self, filepath: str, snapshot_id: Optional[str] = None):
        """保存快照到文件"""
        if snapshot_id is None:
            snapshot_id = self.current_snapshot_id
        
        if snapshot_id not in self.snapshots:
            print(f"错误: 快照 {snapshot_id} 不存在")
            return False
        
        snapshot = self.snapshots[snapshot_id]
        
        try:
            # 使用 pickle 序列化
            with open(filepath, 'wb') as f:
                pickle.dump(snapshot.to_dict(), f)
            print(f"✓ 快照已保存: {filepath}")
            return True
        except Exception as e:
            print(f"✗ 保存快照失败: {e}")
            return False
    
    def load_from_file(self, filepath: str) -> Optional[str]:
        """从文件加载快照"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            # 重建快照对象
            snapshot = FullSystemSnapshot(
                timestamp=data['timestamp'],
                version=data['version'],
                model_state=data['model_state'],
                optimizer_state=data['optimizer_state'],
                python_rng_state=data['python_rng_state'],
                numpy_rng_state=data['numpy_rng_state'],
                torch_rng_state=torch.tensor(data['torch_rng_state']),
                torch_cuda_rng_state=data['torch_cuda_rng_state'],
                runtime_state=data['runtime_state'],
                metadata=data['metadata'],
            )
            
            snapshot_id = data['metadata'].get('name', f"loaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            self.snapshots[snapshot_id] = snapshot
            
            print(f"✓ 快照已加载: {snapshot_id}")
            return snapshot_id
            
        except Exception as e:
            print(f"✗ 加载快照失败: {e}")
            return None
    
    def list_snapshots(self) -> Dict[str, str]:
        """列出所有快照"""
        return {
            sid: f"{s.timestamp} - {s.metadata.get('description', 'No description')}"
            for sid, s in self.snapshots.items()
        }
    
    def delete_snapshot(self, snapshot_id: str) -> bool:
        """删除快照"""
        if snapshot_id in self.snapshots:
            del self.snapshots[snapshot_id]
            if self.current_snapshot_id == snapshot_id:
                self.current_snapshot_id = None
            print(f"✓ 快照已删除: {snapshot_id}")
            return True
        return False


# ==================== 便捷函数 ====================

def create_full_snapshot(model: nn.Module, optimizer: torch.optim.Optimizer = None, name: str = "baseline") -> FullSystemSnapshot:
    """
    创建完整快照的便捷函数
    
    Args:
        model: 模型
        optimizer: 优化器
        name: 快照名称
        
    Returns:
        快照对象
    """
    manager = FullRollbackManager(model, optimizer)
    snapshot_id = manager.create_snapshot(name, {'description': f'Full snapshot: {name}'})
    return manager.snapshots[snapshot_id]


def restore_full_snapshot(model: nn.Module, snapshot: FullSystemSnapshot, optimizer: torch.optim.Optimizer = None) -> bool:
    """
    恢复快照的便捷函数
    
    Args:
        model: 模型
        snapshot: 快照对象
        optimizer: 优化器
        
    Returns:
        是否成功
    """
    manager = FullRollbackManager(model, optimizer)
    # 临时存储快照
    temp_id = "temp_restore"
    manager.snapshots[temp_id] = snapshot
    return manager.restore_snapshot(temp_id)


# ==================== 测试 ====================

def test_full_snapshot():
    """测试完整快照"""
    print("\n" + "="*70)
    print("测试完整 Rollback 快照")
    print("="*70)
    
    # 创建模型和优化器
    config = NativeTinyConfig()
    model = NativeBackboneTinyV1(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # 创建管理器
    manager = FullRollbackManager(model, optimizer)
    
    # 设置随机种子
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # 记录基线输出
    model.eval()
    test_input = torch.randint(0, 10000, (1, 50))
    with torch.no_grad():
        baseline_output = model(test_input)
        baseline_writeback = baseline_output['writeback_probs'].clone()
    
    print(f"\n基线 writeback: {baseline_writeback[0]}")
    
    # 创建快照
    snapshot_id = manager.create_snapshot("baseline", {'description': 'Initial state'})
    
    # 模拟训练
    print("\n模拟训练...")
    model.train()
    for _ in range(3):
        optimizer.zero_grad()
        output = model(test_input)
        loss = output['gap_logits'].mean()
        loss.backward()
        optimizer.step()
    
    # 检查训练后状态
    model.eval()
    with torch.no_grad():
        trained_output = model(test_input)
        trained_writeback = trained_output['writeback_probs']
    
    print(f"训练后 writeback: {trained_writeback[0]}")
    
    # 恢复快照
    success = manager.restore_snapshot(snapshot_id)
    
    # 检查恢复后状态
    model.eval()
    with torch.no_grad():
        restored_output = model(test_input)
        restored_writeback = restored_output['writeback_probs']
    
    print(f"恢复后 writeback: {restored_writeback[0]}")
    
    # 验证
    match = torch.allclose(baseline_writeback, restored_writeback, atol=1e-6)
    print(f"\n验证结果: {'✓ 完全恢复' if match else '✗ 恢复不完全'}")
    
    # 保存到文件测试
    manager.save_to_file("test_snapshot.pkl", snapshot_id)
    
    # 加载测试
    loaded_id = manager.load_from_file("test_snapshot.pkl")
    
    print("\n测试完成")
    
    return match


if __name__ == "__main__":
    from phase19_native_backbone.native_backbone_tiny_v1 import NativeTinyConfig
    test_full_snapshot()
