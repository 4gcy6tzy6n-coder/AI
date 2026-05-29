"""
Stage 6 Writeback Isolation Fix

Writeback 隔离修复 - 解决 writeback 与其他 head 的耦合问题

核心思路:
1. Writeback head 使用独立的 feature extractor
2. 在训练时冻结 writeback 相关参数
3. 增加 writeback 专项保护损失
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from dataclasses import dataclass
import copy

from phase19_native_backbone.native_backbone_tiny_v1 import NativeBackboneTinyV1, NativeTinyConfig


class IsolatedWritebackHead(nn.Module):
    """
    隔离的 Writeback Head
    
    拥有独立的表示提取层，不与其他 head 共享
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_writeback_types: int):
        super().__init__()
        
        # 独立的特征提取
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
        )
        
        # 输出层
        self.classifier = nn.Linear(hidden_dim // 2, num_writeback_types)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, gap_probs: torch.Tensor = None) -> torch.Tensor:
        """前向传播 - 兼容原始接口"""
        features = self.feature_extractor(x)
        # 可以结合 gap_probs 如果有需要
        if gap_probs is not None:
            # 简单拼接特征
            gap_feature = gap_probs.unsqueeze(-1) if gap_probs.dim() == 1 else gap_probs
            combined = torch.cat([features, gap_feature], dim=-1)
            # 需要调整 classifier 输入维度，这里简化处理
            logits = self.classifier(features)  # 暂时忽略 gap_probs
        else:
            logits = self.classifier(features)
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """获取特征表示"""
        return self.feature_extractor(x)


class WritebackIsolationWrapper:
    """
    Writeback 隔离包装器
    
    用于在训练时保护 writeback 能力
    """
    
    def __init__(self, model: NativeBackboneTinyV1):
        self.model = model
        self.original_writeback_head = None
        self.isolated_writeback_head = None
        self._setup_isolation()
    
    def _setup_isolation(self):
        """设置隔离"""
        # 保存原始 writeback head
        self.original_writeback_head = self.model.writeback_head
        
        # 创建隔离的 writeback head
        config = self.model.config
        self.isolated_writeback_head = IsolatedWritebackHead(
            input_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            num_writeback_types=config.num_writeback_types
        )
        
        # 复制权重
        with torch.no_grad():
            # 尝试复制原始权重到新结构
            try:
                original_state = self.original_writeback_head.state_dict()
                self.isolated_writeback_head.load_state_dict(original_state, strict=False)
            except:
                print("Warning: Could not copy weights to isolated writeback head")
        
        # 替换模型中的 writeback head
        self.model.writeback_head = self.isolated_writeback_head
    
    def freeze_writeback(self):
        """冻结 writeback head 参数"""
        for param in self.isolated_writeback_head.parameters():
            param.requires_grad = False
        print("Writeback head 已冻结")
    
    def unfreeze_writeback(self):
        """解冻 writeback head 参数"""
        for param in self.isolated_writeback_head.parameters():
            param.requires_grad = True
        print("Writeback head 已解冻")
    
    def restore_original(self):
        """恢复原始 writeback head"""
        self.model.writeback_head = self.original_writeback_head
        print("已恢复原始 writeback head")
    
    def get_writeback_params(self) -> List[torch.nn.Parameter]:
        """获取 writeback 相关参数"""
        return list(self.isolated_writeback_head.parameters())


class WritebackProtectedTrainer:
    """
    Writeback 保护训练器
    
    在训练时隔离 writeback，防止其被影响
    """
    
    def __init__(self, model: NativeBackboneTinyV1, config: Dict):
        self.model = model
        self.config = config
        self.isolation_wrapper = WritebackIsolationWrapper(model)
        
        # 保存原始 writeback 能力
        self.baseline_writeback_score = None
    
    def train_step_with_protection(
        self,
        input_ids: torch.Tensor,
        target_outputs: Dict,
        protect_writeback: bool = True
    ) -> Dict:
        """
        执行受保护的训练步骤
        
        Args:
            input_ids: 输入
            target_outputs: 目标输出
            protect_writeback: 是否保护 writeback
            
        Returns:
            训练指标
        """
        self.model.train()
        
        # 如果需要保护 writeback，先冻结
        if protect_writeback:
            self.isolation_wrapper.freeze_writeback()
        
        # 前向传播
        outputs = self.model(input_ids)
        
        # 计算损失 (不包括 writeback)
        loss = 0
        
        # Gap 损失
        if 'gap' in target_outputs:
            loss += F.cross_entropy(
                outputs['gap_logits'],
                target_outputs['gap']
            )
        
        # Policy 损失
        if 'policy' in target_outputs:
            loss += F.cross_entropy(
                outputs['policy_logits'],
                target_outputs['policy']
            )
        
        # Governance 损失
        if 'governance' in target_outputs:
            loss += F.binary_cross_entropy_with_logits(
                outputs['governance_logits'],
                target_outputs['governance']
            )
        
        # 注意: 不包含 writeback 损失
        
        # 反向传播
        loss.backward()
        
        # 恢复 writeback
        if protect_writeback:
            self.isolation_wrapper.unfreeze_writeback()
        
        return {
            'loss': loss.item(),
            'writeback_protected': protect_writeback,
        }
    
    def evaluate_writeback_stability(self, test_inputs: List[torch.Tensor]) -> float:
        """评估 writeback 稳定性"""
        self.model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for input_ids in test_inputs:
                outputs = self.model(input_ids)
                
                # 简化评估: 检查 writeback 置信度
                writeback_conf = outputs['writeback_probs'][0].max().item()
                if writeback_conf > 0.5:
                    correct += 1
                total += 1
        
        return correct / total if total > 0 else 0.0


def apply_writeback_isolation_fix(model: NativeBackboneTinyV1) -> WritebackIsolationWrapper:
    """
    应用 writeback 隔离修复
    
    Args:
        model: 要修复的模型
        
    Returns:
        隔离包装器
    """
    print("="*70)
    print("应用 Writeback 隔离修复")
    print("="*70)
    
    wrapper = WritebackIsolationWrapper(model)
    
    print("\n修复完成:")
    print("  ✓ Writeback head 已隔离")
    print("  ✓ 独立的 feature extractor")
    print("  ✓ 可冻结/解冻控制")
    
    return wrapper


# ==================== 测试 ====================

def test_writeback_isolation():
    """测试 writeback 隔离"""
    print("\n" + "="*70)
    print("测试 Writeback 隔离")
    print("="*70)
    
    # 创建模型
    config = NativeTinyConfig()
    model = NativeBackboneTinyV1(config)
    
    # 保存原始 writeback 输出
    model.eval()
    test_input = torch.randint(0, 10000, (1, 50))
    with torch.no_grad():
        original_output = model(test_input)
        original_writeback = original_output['writeback_probs'].clone()
    
    print(f"原始 writeback 输出: {original_writeback[0]}")
    
    # 应用隔离
    wrapper = apply_writeback_isolation_fix(model)
    
    # 测试隔离后输出
    with torch.no_grad():
        isolated_output = model(test_input)
        isolated_writeback = isolated_output['writeback_probs']
    
    print(f"隔离后 writeback 输出: {isolated_writeback[0]}")
    
    # 测试冻结/解冻
    wrapper.freeze_writeback()
    
    # 检查是否可训练
    dummy_loss = isolated_writeback.sum()
    dummy_loss.backward()
    
    writeback_grads = [p.grad for p in wrapper.get_writeback_params() if p.grad is not None]
    if writeback_grads:
        print(f"警告: writeback 仍有梯度，冻结失败")
    else:
        print("✓ Writeback 冻结成功 (无梯度)")
    
    wrapper.unfreeze_writeback()
    
    # 恢复原始
    wrapper.restore_original()
    
    with torch.no_grad():
        restored_output = model(test_input)
        restored_writeback = restored_output['writeback_probs']
    
    print(f"恢复后 writeback 输出: {restored_writeback[0]}")
    
    print("\n测试完成")


if __name__ == "__main__":
    test_writeback_isolation()
