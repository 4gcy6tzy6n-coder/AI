"""
Training Core Module - 训练核心模块

第五阶段新增：
- review_manager: 强审查节点
- verifier: 验证节点
- case_builder: 训练案例构建
- noise_injector: 混淆注入
"""

from .review_manager import ReviewManager, ReviewResult, ReviewCheck
from .verifier import Verifier, VerificationResult

__all__ = [
    "ReviewManager",
    "ReviewResult",
    "ReviewCheck",
    "Verifier",
    "VerificationResult"
]
