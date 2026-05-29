"""
Backbone-level Feature Guard

基于正确来源拆解实验的新设计:
- Backbone训练是writeback漂移的主要来源
- 需要选择性保护writeback-sensitive特征子空间
- 允许backbone其他部分自由学习主任务

两种实现:
1. FeaturePreservationGuard: 全局特征保持
2. SubspaceProjectionGuard: 子空间投影保护 (推荐)
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Set
import copy


class FeatureGuardConfig:
    """Feature Guard 配置"""
    def __init__(
        self,
        mode: str = "subspace",  # "preservation" or "subspace"
        anchor_samples: int = 30,
        beta: float = 0.05,  # 特征保护权重
        top_k_dims: int = 64,  # 子空间维度 (用于subspace模式)
        distance_metric: str = "cosine",  # "cosine" or "l2"
    ):
        self.mode = mode
        self.anchor_samples = anchor_samples
        self.beta = beta
        self.top_k_dims = top_k_dims
        self.distance_metric = distance_metric


class BackboneFeatureGuard:
    """
    Backbone特征保护基类
    
    核心功能:
    1. 保存训练前参考特征
    2. 训练中计算特征保持损失
    3. 约束writeback-sensitive特征子空间
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        config: FeatureGuardConfig,
    ):
        self.model = model
        self.config = config
        
        # 参考特征 (训练前保存)
        self.ref_features: Optional[torch.Tensor] = None
        self.anchor_inputs: Optional[List[torch.Tensor]] = None
        
        # 子空间投影 (用于subspace模式)
        self.sensitive_dims: Optional[Set[int]] = None
        self.projection_mask: Optional[torch.Tensor] = None
        
    def setup_anchor_samples(self, num_samples: int = None, seed: int = 42):
        """
        设置anchor样本用于特征保持
        
        这些样本固定，用于计算特征漂移
        """
        if num_samples is None:
            num_samples = self.config.anchor_samples
        
        torch.manual_seed(seed)
        self.anchor_inputs = [
            torch.randint(0, 10000, (1, 50))
            for _ in range(num_samples)
        ]
        print(f"[FeatureGuard] Anchor samples generated: {num_samples}")
    
    def capture_reference_features(self):
        """
        捕获训练前的参考特征
        
        在训练开始前调用，保存baseline特征
        """
        if self.anchor_inputs is None:
            self.setup_anchor_samples()
        
        self.model.eval()
        with torch.no_grad():
            ref_features_list = []
            for input_ids in self.anchor_inputs:
                # 提取backbone特征 (writeback head的输入)
                features = self._extract_backbone_features(input_ids)
                ref_features_list.append(features)
            
            self.ref_features = torch.cat(ref_features_list, dim=0)
        
        print(f"[FeatureGuard] Reference features captured: shape={self.ref_features.shape}")
        
        # 如果是subspace模式，识别敏感维度
        if self.config.mode == "subspace":
            self._identify_sensitive_dimensions()
    
    def _extract_backbone_features(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        提取backbone特征 (writeback head的输入)
        
        注意: 此方法必须在梯度上下文中调用，以支持反向传播
        """
        outputs = self.model(input_ids)
        
        # 尝试多种可能的特征提取方式
        if 'backbone_features' in outputs:
            return outputs['backbone_features']
        elif 'hidden_states' in outputs:
            return outputs['hidden_states'][-1]  # 最后一层
        else:
            # 默认: 返回模型内部特征
            # 这里假设可以通过forward hook获取
            return self._extract_via_hook(input_ids)
    
    def _extract_via_hook(self, input_ids: torch.Tensor) -> torch.Tensor:
        """通过forward hook提取特征 (支持梯度)"""
        features = []
        
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                features.append(output)  # 不移除梯度
        
        # 注册hook到最后一层
        handles = []
        for name, module in self.model.named_modules():
            if 'encoder' in name.lower() or 'transformer' in name.lower():
                if hasattr(module, 'forward'):
                    handle = module.register_forward_hook(hook_fn)
                    handles.append(handle)
                    break
        
        # 前向传播 (允许梯度)
        _ = self.model(input_ids)
        
        # 移除hooks
        for handle in handles:
            handle.remove()
        
        if features:
            return features[0]
        else:
            #  fallback: 返回输入embedding
            return self.model.embeddings(input_ids) if hasattr(self.model, 'embeddings') else input_ids.float()
    
    def _identify_sensitive_dimensions(self):
        """
        识别writeback-sensitive维度
        
        简化版: 基于特征方差选择维度
        方差大的维度通常对输出影响更大
        """
        print("[FeatureGuard] Identifying writeback-sensitive dimensions...")
        
        if self.ref_features is None:
            print("  Warning: No reference features, using all dimensions")
            return
        
        # 基于特征方差选择维度
        feature_var = self.ref_features.var(dim=0)  # 各维度方差
        
        # 选择top-k高方差维度
        top_k = min(self.config.top_k_dims, len(feature_var))
        top_dims = torch.topk(feature_var, top_k).indices.tolist()
        self.sensitive_dims = set(top_dims)
        
        # 创建投影mask
        self.projection_mask = torch.zeros(len(feature_var))
        self.projection_mask[list(self.sensitive_dims)] = 1.0
        
        print(f"  Selected {len(self.sensitive_dims)} high-variance dimensions")
        print(f"  Top 10 dims: {sorted(list(self.sensitive_dims))[:10]}")
    
    def compute_feature_guard_loss(self) -> torch.Tensor:
        """
        计算特征保护损失
        
        Returns:
            特征保持损失
        """
        if self.ref_features is None:
            raise RuntimeError("Must call capture_reference_features() before training")
        
        # 获取当前特征 (允许梯度)
        current_features_list = []
        for input_ids in self.anchor_inputs:
            features = self._extract_backbone_features(input_ids)
            current_features_list.append(features)
        
        current_features = torch.cat(current_features_list, dim=0)
        
        # 计算损失
        if self.config.mode == "subspace" and self.projection_mask is not None:
            # 子空间模式: 只约束敏感维度
            mask = self.projection_mask.to(current_features.device)
            ref_masked = self.ref_features * mask
            curr_masked = current_features * mask
            
            if self.config.distance_metric == "cosine":
                loss = 1 - F.cosine_similarity(curr_masked.flatten(), ref_masked.flatten(), dim=0)
            else:
                loss = F.mse_loss(curr_masked, ref_masked)
        else:
            # 全局保持模式
            if self.config.distance_metric == "cosine":
                loss = 1 - F.cosine_similarity(current_features.flatten(), self.ref_features.flatten(), dim=0)
            else:
                loss = F.mse_loss(current_features, self.ref_features)
        
        return loss * self.config.beta
    
    def get_diagnostics(self) -> Dict:
        """获取诊断信息"""
        return {
            'mode': self.config.mode,
            'anchor_samples': len(self.anchor_inputs) if self.anchor_inputs else 0,
            'sensitive_dims': len(self.sensitive_dims) if self.sensitive_dims else 0,
            'beta': self.config.beta,
            'reference_captured': self.ref_features is not None,
        }


class FeaturePreservationGuard(BackboneFeatureGuard):
    """
    方案A: 全局特征保持
    
    约束整个特征向量不要偏离参考值
    """
    
    def __init__(self, model: torch.nn.Module, beta: float = 0.05):
        config = FeatureGuardConfig(mode="preservation", beta=beta)
        super().__init__(model, config)


class SubspaceProjectionGuard(BackboneFeatureGuard):
    """
    方案B: 子空间投影保护 (推荐)
    
    只约束writeback-sensitive维度
    更精准，对主任务影响更小
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        beta: float = 0.05,
        top_k_dims: int = 64,
    ):
        config = FeatureGuardConfig(
            mode="subspace",
            beta=beta,
            top_k_dims=top_k_dims,
        )
        super().__init__(model, config)


# 便捷构建函数
def build_feature_preservation_guard(model: torch.nn.Module, beta: float = 0.05) -> FeaturePreservationGuard:
    """构建全局特征保持guard"""
    return FeaturePreservationGuard(model, beta)


def build_subspace_projection_guard(
    model: torch.nn.Module,
    beta: float = 0.05,
    top_k_dims: int = 64,
) -> SubspaceProjectionGuard:
    """构建子空间投影guard (推荐)"""
    return SubspaceProjectionGuard(model, beta, top_k_dims)


# 测试代码
if __name__ == "__main__":
    print("="*70)
    print("Backbone Feature Guard 测试")
    print("="*70)
    
    # 创建一个简单模型用于测试
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = torch.nn.Embedding(10000, 128)
            self.encoder = torch.nn.Linear(128, 128)
            self.writeback_head = torch.nn.Linear(128, 2)
        
        def forward(self, input_ids):
            x = self.embeddings(input_ids).mean(dim=1)
            features = self.encoder(x)
            writeback_logits = self.writeback_head(features)
            return {
                'writeback_logits': writeback_logits,
                'backbone_features': features,
            }
    
    model = DummyModel()
    
    # 测试 SubspaceProjectionGuard
    print("\n测试 SubspaceProjectionGuard:")
    guard = build_subspace_projection_guard(model, beta=0.05, top_k_dims=32)
    guard.capture_reference_features()
    
    # 模拟训练
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for step in range(3):
        optimizer.zero_grad()
        
        # 主损失
        input_ids = torch.randint(0, 10000, (4, 50))
        outputs = model(input_ids)
        main_loss = outputs['writeback_logits'].mean()
        
        # 特征保护损失
        guard_loss = guard.compute_feature_guard_loss()
        
        total_loss = main_loss + guard_loss
        total_loss.backward()
        optimizer.step()
        
        print(f"  Step {step+1}: main_loss={main_loss.item():.4f}, guard_loss={guard_loss.item():.6f}")
    
    print("\n✓ 测试完成")
