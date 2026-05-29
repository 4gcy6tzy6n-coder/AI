"""
Stage 7 Writeback 保护机制 V2

可插拔策略框架，支持 A/B/C/D 四种方案切换
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum


class WritebackGuardMode(Enum):
    """Writeback 保护模式"""
    FEATURE_ISOLATION = "feature_isolation"      # 方案 A
    GRADIENT_MASK = "gradient_mask"              # 方案 B
    INDEPENDENT_OPTIMIZER = "independent_optimizer"  # 方案 C
    HYBRID = "hybrid"                            # 方案 D (组合)
    FREEZE = "freeze"                            # 方案 E: 完全冻结
    LOW_LR_CLIP = "low_lr_clip"                  # 方案 F: 低学习率+梯度裁剪
    DELTA_PENALTY = "delta_penalty"              # 方案 G: 参数变化惩罚


@dataclass
class WritebackGuardConfig:
    """Writeback 保护配置"""
    mode: WritebackGuardMode = WritebackGuardMode.HYBRID
    
    # 特征隔离配置
    isolate_layer: int = -2  # 从第 N-2 层开始隔离
    feature_dim: int = 768   # 特征维度
    
    # 梯度屏蔽配置
    shield_backbone: bool = True  # 是否屏蔽 backbone 梯度
    
    # 独立优化器配置
    writeback_lr: float = 1e-5
    main_lr: float = 1e-5
    
    # 低学习率+裁剪配置
    low_lr: float = 1e-6  # writeback head 使用更低学习率
    grad_clip_value: float = 0.01  # 梯度裁剪阈值
    
    # 参数变化惩罚配置
    delta_penalty_coef: float = 0.1  # 惩罚系数
    
    # 混合模式配置
    hybrid_components: List[str] = field(default_factory=lambda: ["feature_isolation", "gradient_mask"])


class FeatureIsolator(nn.Module):
    """
    方案 A: 特征隔离
    
    为 writeback head 创建独立特征提取路径
    """
    
    def __init__(self, input_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = input_dim // 2
        
        self.isolation_transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
            nn.LayerNorm(input_dim),
        )
        
        # 初始化为单位映射附近
        with torch.no_grad():
            self.isolation_transform[-2].weight.fill_(0)
            self.isolation_transform[-2].bias.fill_(0)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        应用特征隔离变换
        
        Args:
            features: 原始特征 [batch, dim]
            
        Returns:
            隔离后的特征 [batch, dim]
        """
        # 残差连接，保持原始信息
        transformed = self.isolation_transform(features)
        return features + 0.1 * transformed  # 小幅度调整


class GradientShield:
    """
    方案 B: 梯度屏蔽
    
    在反向传播时屏蔽其他 head 对 writeback path 的影响
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.writeback_param_ids: set = set()
        self.enabled: bool = False
        
    def register_writeback_params(self, writeback_module: nn.Module):
        """注册 writeback 专用参数"""
        for param in writeback_module.parameters():
            self.writeback_param_ids.add(id(param))
        print(f"[GradientShield] 注册 {len(self.writeback_param_ids)} 个 writeback 参数")
    
    def enable(self):
        """启用梯度屏蔽"""
        self.enabled = True
        # 注册反向传播钩子
        self._register_hooks()
        print("[GradientShield] 已启用")
    
    def disable(self):
        """禁用梯度屏蔽"""
        self.enabled = False
        self._remove_hooks()
        print("[GradientShield] 已禁用")
    
    def _register_hooks(self):
        """注册梯度屏蔽钩子"""
        self.hooks = []
        for name, param in self.model.named_parameters():
            if id(param) not in self.writeback_param_ids:
                hook = param.register_hook(self._shield_hook)
                self.hooks.append(hook)
    
    def _remove_hooks(self):
        """移除梯度屏蔽钩子"""
        for hook in getattr(self, 'hooks', []):
            hook.remove()
        self.hooks = []
    
    def _shield_hook(self, grad: torch.Tensor) -> torch.Tensor:
        """梯度屏蔽钩子函数"""
        if not self.enabled:
            return grad
        # 屏蔽非 writeback 参数的梯度
        return torch.zeros_like(grad)


class IndependentOptimizerManager:
    """
    方案 C: 独立优化器
    
    Writeback head 使用完全独立的优化器
    """
    
    def __init__(self, 
                 writeback_params: List[torch.nn.Parameter],
                 main_params: List[torch.nn.Parameter],
                 writeback_lr: float = 1e-5,
                 main_lr: float = 1e-5):
        
        self.writeback_optimizer = torch.optim.AdamW(
            writeback_params,
            lr=writeback_lr,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
        self.main_optimizer = torch.optim.AdamW(
            main_params,
            lr=main_lr,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
        print(f"[IndependentOptimizer] Writeback 优化器: {len(writeback_params)} 参数")
        print(f"[IndependentOptimizer] Main 优化器: {len(main_params)} 参数")
    
    def step_writeback(self, loss: torch.Tensor):
        """优化 writeback"""
        self.writeback_optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.writeback_optimizer.step()
    
    def step_main(self, loss: torch.Tensor):
        """优化 main"""
        self.main_optimizer.zero_grad()
        loss.backward()
        self.main_optimizer.step()
    
    def step_both(self, writeback_loss: torch.Tensor, main_loss: torch.Tensor):
        """分别优化两者"""
        self.step_writeback(writeback_loss)
        self.step_main(main_loss)
    
    def zero_grad_all(self):
        """清零所有梯度"""
        self.writeback_optimizer.zero_grad()
        self.main_optimizer.zero_grad()


class WritebackGuardV2:
    """
    Stage 7 Writeback 综合保护机制 V2
    
    可插拔策略框架，支持模式切换
    """
    
    def __init__(self, 
                 model: nn.Module,
                 writeback_head: nn.Module,
                 config: Optional[WritebackGuardConfig] = None):
        
        self.model = model
        self.writeback_head = writeback_head
        self.config = config or WritebackGuardConfig()
        
        # 组件
        self.feature_isolator: Optional[FeatureIsolator] = None
        self.gradient_shield: Optional[GradientShield] = None
        self.independent_optimizer: Optional[IndependentOptimizerManager] = None
        
        # 初始化
        self._build_guard()
    
    def _build_guard(self):
        """根据配置构建保护机制"""
        mode = self.config.mode
        
        if mode == WritebackGuardMode.FEATURE_ISOLATION:
            self._build_feature_isolation()
        elif mode == WritebackGuardMode.GRADIENT_MASK:
            self._build_gradient_mask()
        elif mode == WritebackGuardMode.INDEPENDENT_OPTIMIZER:
            self._build_independent_optimizer()
        elif mode == WritebackGuardMode.HYBRID:
            self._build_hybrid()
        
        print(f"[WritebackGuardV2] 已构建模式: {mode.value}")
    
    def _build_feature_isolation(self):
        """构建特征隔离"""
        self.feature_isolator = FeatureIsolator(
            input_dim=self.config.feature_dim,
        )
        print("[WritebackGuardV2] 特征隔离已启用")
    
    def _build_gradient_mask(self):
        """构建梯度屏蔽"""
        self.gradient_shield = GradientShield(self.model)
        self.gradient_shield.register_writeback_params(self.writeback_head)
        print("[WritebackGuardV2] 梯度屏蔽已启用")
    
    def _build_independent_optimizer(self):
        """构建独立优化器"""
        writeback_params = list(self.writeback_head.parameters())
        main_params = [p for p in self.model.parameters() 
                      if id(p) not in [id(wp) for wp in writeback_params]]
        
        self.independent_optimizer = IndependentOptimizerManager(
            writeback_params=writeback_params,
            main_params=main_params,
            writeback_lr=self.config.writeback_lr,
            main_lr=self.config.main_lr,
        )
        print("[WritebackGuardV2] 独立优化器已启用")
    
    def _build_hybrid(self):
        """构建混合模式"""
        components = self.config.hybrid_components
        
        if "feature_isolation" in components:
            self._build_feature_isolation()
        
        if "gradient_mask" in components:
            self._build_gradient_mask()
        
        if "independent_optimizer" in components:
            self._build_independent_optimizer()
        
        print(f"[WritebackGuardV2] 混合模式组件: {components}")
    
    def apply_feature_isolation(self, features: torch.Tensor) -> torch.Tensor:
        """
        应用特征隔离
        
        Args:
            features: 输入特征
            
        Returns:
            隔离后的特征
        """
        if self.feature_isolator is not None:
            return self.feature_isolator(features)
        return features
    
    def enable_gradient_shield(self):
        """启用梯度屏蔽"""
        if self.gradient_shield is not None:
            self.gradient_shield.enable()
    
    def disable_gradient_shield(self):
        """禁用梯度屏蔽"""
        if self.gradient_shield is not None:
            self.gradient_shield.disable()
    
    def get_optimizers(self) -> Optional[IndependentOptimizerManager]:
        """获取独立优化器"""
        return self.independent_optimizer
    
    def freeze_writeback(self):
        """冻结 writeback head 参数"""
        for param in self.writeback_head.parameters():
            param.requires_grad = False
        print("[WritebackGuardV2] Writeback head 已冻结")
    
    def unfreeze_writeback(self):
        """解冻 writeback head 参数"""
        for param in self.writeback_head.parameters():
            param.requires_grad = True
        print("[WritebackGuardV2] Writeback head 已解冻")
    
    def apply_low_lr_clip(self, optimizer: torch.optim.Optimizer):
        """
        应用低学习率 + 梯度裁剪
        
        修改优化器参数组，为 writeback head 设置更低学习率
        """
        # 获取 writeback 参数
        writeback_params = set(id(p) for p in self.writeback_head.parameters())
        
        # 重新组织参数组
        param_groups = optimizer.param_groups
        
        # 分离 writeback 和 main 参数
        wb_params = []
        main_params = []
        
        for group in param_groups:
            for p in group['params']:
                if id(p) in writeback_params:
                    wb_params.append(p)
                else:
                    main_params.append(p)
        
        # 清空原参数组
        optimizer.param_groups.clear()
        
        # 添加 main 参数组 (原学习率)
        if main_params:
            optimizer.add_param_group({'params': main_params})
        
        # 添加 writeback 参数组 (低学习率)
        if wb_params:
            optimizer.add_param_group({
                'params': wb_params,
                'lr': self.config.low_lr,
                'max_grad_norm': self.config.grad_clip_value
            })
        
        print(f"[WritebackGuardV2] 低学习率保护已应用: lr={self.config.low_lr}, clip={self.config.grad_clip_value}")
    
    def compute_delta_penalty(self) -> torch.Tensor:
        """
        计算参数变化惩罚
        
        Returns:
            惩罚项 (标量)
        """
        if not hasattr(self, '_wb_params_ref'):
            # 首次调用，保存参考参数
            self._wb_params_ref = {
                id(p): p.clone().detach()
                for p in self.writeback_head.parameters()
            }
            return torch.tensor(0.0)
        
        # 计算当前参数与参考参数的差异
        penalty = 0.0
        for p in self.writeback_head.parameters():
            ref = self._wb_params_ref.get(id(p))
            if ref is not None:
                penalty += torch.norm(p - ref).item() ** 2
        
        return torch.tensor(penalty * self.config.delta_penalty_coef)
    
    def switch_mode(self, mode: WritebackGuardMode, new_config: Optional[WritebackGuardConfig] = None):
        """
        切换保护模式
        
        用于实验不同策略
        """
        # 清理旧组件
        self.feature_isolator = None
        self.gradient_shield = None
        self.independent_optimizer = None
        
        # 更新配置
        self.config = new_config or WritebackGuardConfig(mode=mode)
        
        # 重新构建
        self._build_guard()
        
        print(f"[WritebackGuardV2] 已切换到模式: {mode.value}")


def build_writeback_guard(
    model: nn.Module,
    writeback_head: nn.Module,
    mode: str = "hybrid",
    config: Optional[Dict[str, Any]] = None
) -> WritebackGuardV2:
    """
    构建 Writeback 保护机制
    
    这是标准入口函数
    
    Args:
        model: 主模型
        writeback_head: writeback head
        mode: 保护模式 ("feature_isolation", "gradient_mask", "independent_optimizer", "hybrid")
        config: 额外配置
        
    Returns:
        WritebackGuardV2 实例
    """
    mode_enum = WritebackGuardMode(mode)
    guard_config = WritebackGuardConfig(mode=mode_enum)
    
    if config:
        for key, value in config.items():
            if hasattr(guard_config, key):
                setattr(guard_config, key, value)
    
    return WritebackGuardV2(model, writeback_head, guard_config)


def run_writeback_protected_step(
    guard: WritebackGuardV2,
    features: torch.Tensor,
    writeback_loss_fn: Callable,
    main_loss_fn: Optional[Callable] = None
) -> Dict[str, torch.Tensor]:
    """
    运行受保护的 writeback 步骤
    
    Args:
        guard: WritebackGuardV2 实例
        features: 输入特征
        writeback_loss_fn: writeback 损失函数
        main_loss_fn: 主损失函数 (可选)
        
    Returns:
        损失字典
    """
    # 应用特征隔离
    isolated_features = guard.apply_feature_isolation(features)
    
    # 计算 writeback 损失
    writeback_loss = writeback_loss_fn(isolated_features)
    
    # 根据模式执行优化
    if guard.independent_optimizer is not None:
        # 使用独立优化器
        if main_loss_fn is not None:
            main_loss = main_loss_fn(features)
            guard.independent_optimizer.step_both(writeback_loss, main_loss)
            return {'writeback': writeback_loss, 'main': main_loss}
        else:
            guard.independent_optimizer.step_writeback(writeback_loss)
            return {'writeback': writeback_loss}
    else:
        # 使用普通反向传播
        writeback_loss.backward()
        return {'writeback': writeback_loss}


# 便捷的工厂函数
def build_feature_isolation_only(model: nn.Module, writeback_head: nn.Module) -> WritebackGuardV2:
    """仅使用特征隔离 (方案 A)"""
    return build_writeback_guard(model, writeback_head, mode="feature_isolation")


def build_gradient_mask_only(model: nn.Module, writeback_head: nn.Module) -> WritebackGuardV2:
    """仅使用梯度屏蔽 (方案 B)"""
    return build_writeback_guard(model, writeback_head, mode="gradient_mask")


def build_independent_optimizer_only(model: nn.Module, writeback_head: nn.Module) -> WritebackGuardV2:
    """仅使用独立优化器 (方案 C)"""
    return build_writeback_guard(model, writeback_head, mode="independent_optimizer")


def build_hybrid_guard(model: nn.Module, writeback_head: nn.Module, 
                       components: List[str] = None) -> WritebackGuardV2:
    """
    使用混合模式 (方案 D)
    
    Args:
        components: 组件列表，如 ["feature_isolation", "gradient_mask"]
    """
    config = {'hybrid_components': components or ["feature_isolation", "gradient_mask"]}
    return build_writeback_guard(model, writeback_head, mode="hybrid", config=config)


if __name__ == "__main__":
    # 测试代码
    print("Stage 7 Writeback Guard V2")
    print("=" * 50)
    
    # 创建测试模型
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 10)
    
    class DummyWritebackHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 1)
    
    model = DummyModel()
    writeback_head = DummyWritebackHead()
    
    # 测试不同模式
    for mode in ["feature_isolation", "gradient_mask", "independent_optimizer", "hybrid"]:
        print(f"\n测试模式: {mode}")
        guard = build_writeback_guard(model, writeback_head, mode=mode)
        print(f"  特征隔离: {guard.feature_isolator is not None}")
        print(f"  梯度屏蔽: {guard.gradient_shield is not None}")
        print(f"  独立优化器: {guard.independent_optimizer is not None}")
