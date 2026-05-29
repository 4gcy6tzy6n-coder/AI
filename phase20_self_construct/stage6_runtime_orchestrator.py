"""
Stage 6 Runtime Orchestrator

全链路运行时编排器 - 将受控自构建机制整合为统一运行主链

核心功能:
1. 整合 Native Backbone + Gap/Policy/Governance/Writeback
2. 候选生成与类型分流
3. 参数晋升与知识库晋升
4. 回滚钩子集成
5. 统一指标收集

使用 6B 配置作为默认
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import copy

from phase19_native_backbone.native_backbone_tiny_v1 import NativeBackboneTinyV1, NativeTinyConfig


# ==================== 配置定义 ====================

@dataclass
class Stage6Config:
    """Stage 6 默认配置 (6B)"""
    # 主干配置
    backbone_path: str = "../phase19_native_backbone/checkpoints/native_128.pt"
    device: str = "cpu"
    
    # 候选生成配置
    num_candidates: int = 5
    min_quality_score: float = 0.70
    
    # 类型分流配置
    param_promotion_threshold: float = 0.90
    kb_promotion_threshold: float = 0.80
    long_term_threshold: float = 0.70
    
    # 参数晋升配置 (6B)
    base_kl_weights: Dict[str, float] = field(default_factory=lambda: {
        'gap': 0.30,
        'policy': 0.30,
        'governance': 0.30,
        'writeback': 0.38,
    })
    learning_rate: float = 1.0e-5
    replay_ratio: float = 0.35
    step1_max_change: float = 0.003
    step2_max_change: float = 0.008
    num_epochs: int = 3
    kl_threshold_high: float = 0.05
    kl_threshold_low: float = 0.04
    kl_adjust_up: float = 1.05
    kl_adjust_down: float = 0.98
    
    # 回滚配置
    old_ability_threshold: float = 0.10
    writeback_threshold: float = 0.05
    auto_rollback: bool = True


# ==================== 数据类定义 ====================

@dataclass
class Candidate:
    """候选"""
    candidate_id: str
    candidate_type: str  # RELATION/EXPLANATION/RULE/PATTERN
    content: str
    entities: Dict
    metadata: Dict
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class BackboneOutput:
    """主干输出"""
    gap_logits: torch.Tensor
    gap_probs: torch.Tensor
    policy_logits: torch.Tensor
    policy_probs: torch.Tensor
    governance_logits: torch.Tensor
    governance_probs: torch.Tensor
    writeback_logits: torch.Tensor
    writeback_probs: torch.Tensor
    
    def get_decisions(self) -> Dict:
        """获取决策结果"""
        return {
            'gap': self.gap_probs.argmax(dim=-1).item(),
            'policy': self.policy_probs.argmax(dim=-1).item(),
            'governance': self.governance_probs.argmax(dim=-1).item(),
            'writeback': self.writeback_probs.argmax(dim=-1).item(),
        }


@dataclass
class StepMetrics:
    """单步指标"""
    step_number: int
    timestamp: str
    target_improvement: float
    old_ability_drop: float
    writeback_change: float
    retrieval_change: float
    policy_change: float
    governance_change: float
    kl_weights: Dict[str, float]
    rollback_triggered: bool = False
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PromotionResult:
    """晋升结果"""
    success: bool
    target_improvement: float
    old_ability_drop: float
    writeback_change: float
    steps: List[StepMetrics]
    final_kl_weights: Dict[str, float]
    rollback_performed: bool = False
    error_message: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'success': self.success,
            'target_improvement': self.target_improvement,
            'old_ability_drop': self.old_ability_drop,
            'writeback_change': self.writeback_change,
            'steps': [s.to_dict() for s in self.steps],
            'final_kl_weights': self.final_kl_weights,
            'rollback_performed': self.rollback_performed,
            'error_message': self.error_message,
        }


# ==================== 核心模块 ====================

class BackboneWrapper:
    """主干包装器"""
    
    def __init__(self, config: Stage6Config):
        self.config = config
        model_config = NativeTinyConfig()
        self.model = NativeBackboneTinyV1(model_config)
        self.model.load_state_dict(torch.load(config.backbone_path, map_location=config.device))
        self.model.eval()
    
    def forward(self, input_ids: torch.Tensor) -> BackboneOutput:
        """前向推理"""
        with torch.no_grad():
            outputs = self.model(input_ids)
        return BackboneOutput(
            gap_logits=outputs['gap_logits'],
            gap_probs=outputs['gap_probs'],
            policy_logits=outputs['policy_logits'],
            policy_probs=outputs['policy_probs'],
            governance_logits=outputs['governance_logits'],
            governance_probs=outputs['governance_probs'],
            writeback_logits=outputs['writeback_logits'],
            writeback_probs=outputs['writeback_probs'],
        )
    
    def get_model(self) -> NativeBackboneTinyV1:
        """获取模型实例"""
        return self.model


class CandidateGenerator:
    """候选生成器 (简化版)"""
    
    def __init__(self, config: Stage6Config):
        self.config = config
        self.counter = 0
    
    def generate(
        self,
        input_context: str,
        backbone_output: BackboneOutput,
        num_candidates: int = None
    ) -> List[Candidate]:
        """生成候选"""
        if num_candidates is None:
            num_candidates = self.config.num_candidates
        
        candidates = []
        decisions = backbone_output.get_decisions()
        
        # 基于决策生成候选
        if decisions['gap'] == 1:  # 需要新知识
            for i in range(num_candidates):
                self.counter += 1
                candidate = Candidate(
                    candidate_id=f"cand_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.counter}",
                    candidate_type='RELATION' if i % 2 == 0 else 'EXPLANATION',
                    content=f"Generated candidate from: {input_context[:50]}...",
                    entities={'a': 'entity_a', 'b': 'entity_b'},
                    metadata={
                        'quality_score': 0.85 + (i * 0.02),
                        'source': 'runtime_generation',
                        'timestamp': datetime.now().isoformat(),
                    }
                )
                candidates.append(candidate)
        
        return candidates


class TypeRouter:
    """类型分流器"""
    
    def __init__(self, config: Stage6Config):
        self.config = config
    
    def route(self, candidate: Candidate) -> str:
        """决定候选路径"""
        score = candidate.metadata.get('quality_score', 0)
        cand_type = candidate.candidate_type
        
        # 参数晋升: RELATION + 高分
        if cand_type == 'RELATION' and score >= self.config.param_promotion_threshold:
            return 'PARAM_PROMOTION'
        
        # 知识库晋升: EXPLANATION/RULE + 中高分
        if cand_type in ['EXPLANATION', 'RULE'] and score >= self.config.kb_promotion_threshold:
            return 'KB_PROMOTION'
        
        # 长期候选池
        if score >= self.config.long_term_threshold:
            return 'LONG_TERM'
        
        # 丢弃
        return 'DISCARD'


class RollbackHook:
    """回滚钩子"""
    
    def __init__(self, config: Stage6Config):
        self.config = config
        self.baseline_state = None
        self.baseline_abilities = None
    
    def save_baseline(self, model: NativeBackboneTinyV1, abilities: Dict):
        """保存基线"""
        self.baseline_state = copy.deepcopy(model.state_dict())
        self.baseline_abilities = abilities.copy()
    
    def check_need_rollback(self, current_metrics: Dict) -> bool:
        """检查是否需要回滚"""
        old_drop = current_metrics.get('old_ability_drop', 0)
        writeback_drop = abs(current_metrics.get('writeback_change', 0))
        
        if old_drop > self.config.old_ability_threshold:
            return True
        if writeback_drop > self.config.writeback_threshold:
            return True
        return False
    
    def rollback(self, model: NativeBackboneTinyV1) -> bool:
        """执行回滚"""
        if self.baseline_state is None:
            return False
        
        try:
            model.load_state_dict(self.baseline_state)
            return True
        except Exception as e:
            print(f"Rollback failed: {e}")
            return False


class ParamPromoter:
    """参数晋升模块 - 6B 配置"""
    
    def __init__(self, model: NativeBackboneTinyV1, config: Stage6Config):
        self.model = model
        self.config = config
        self.current_kl_weights = copy.deepcopy(config.base_kl_weights)
        self.optimizer = self._create_optimizer()
    
    def _create_optimizer(self):
        """创建优化器"""
        target_patterns = ['unit_encoder', 'policy_head']
        params = []
        for name, param in self.model.named_parameters():
            if any(p in name for p in target_patterns):
                params.append(param)
        return torch.optim.AdamW(params, lr=self.config.learning_rate)
    
    def _evaluate_abilities(self) -> Dict:
        """评估能力"""
        self.model.eval()
        abilities = {'target': 0, 'retrieval': 0, 'governance': 0, 'policy': 0, 'writeback': 0}
        
        with torch.no_grad():
            for _ in range(50):
                input_ids = torch.randint(0, 10000, (1, 50))
                outputs = self.model(input_ids)
                
                gap_conf = outputs['gap_probs'][0].max().item()
                policy_conf = outputs['policy_probs'][0].max().item()
                if gap_conf > 0.6 and policy_conf > 0.6:
                    abilities['target'] += 1
                if outputs['policy_probs'][0].max().item() > 0.5:
                    abilities['retrieval'] += 1
                    abilities['policy'] += 1
                if outputs['governance_probs'][0].max().item() > 0.5:
                    abilities['governance'] += 1
                if outputs['writeback_probs'][0].max().item() > 0.5:
                    abilities['writeback'] += 1
        
        return {k: v / 50 for k, v in abilities.items()}
    
    def promote(self, candidate: Candidate, step_num: int, 
                baseline_abilities: Dict) -> StepMetrics:
        """执行单步晋升"""
        self.model.train()
        max_change = self.config.step1_max_change if step_num == 1 else self.config.step2_max_change
        
        # 模拟训练 (简化版)
        for epoch in range(self.config.num_epochs):
            self.optimizer.zero_grad()
            
            # 模拟损失
            input_ids = torch.randint(0, 10000, (1, 50))
            outputs = self.model(input_ids)
            
            # 简化的损失计算
            loss = outputs['gap_logits'].mean() + outputs['policy_logits'].mean()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_change)
            self.optimizer.step()
        
        # 评估
        current_abilities = self._evaluate_abilities()
        
        return StepMetrics(
            step_number=step_num,
            timestamp=datetime.now().isoformat(),
            target_improvement=current_abilities['target'] - baseline_abilities['target'],
            old_ability_drop=max(
                abs(current_abilities['retrieval'] - baseline_abilities['retrieval']),
                abs(current_abilities['policy'] - baseline_abilities['policy']),
                abs(current_abilities['governance'] - baseline_abilities['governance']),
            ),
            writeback_change=current_abilities['writeback'] - baseline_abilities['writeback'],
            retrieval_change=current_abilities['retrieval'] - baseline_abilities['retrieval'],
            policy_change=current_abilities['policy'] - baseline_abilities['policy'],
            governance_change=current_abilities['governance'] - baseline_abilities['governance'],
            kl_weights=self.current_kl_weights.copy(),
        )


class KBPromoter:
    """知识库晋升模块 (简化版)"""
    
    def __init__(self, config: Stage6Config):
        self.config = config
        self.kb = []
    
    def promote(self, candidate: Candidate) -> Dict:
        """晋升到知识库"""
        entry = {
            'id': candidate.candidate_id,
            'type': candidate.candidate_type,
            'content': candidate.content,
            'entities': candidate.entities,
            'metadata': candidate.metadata,
            'timestamp': datetime.now().isoformat(),
        }
        self.kb.append(entry)
        return {'success': True, 'kb_id': candidate.candidate_id}


# ==================== 主编排器 ====================

class Stage6Orchestrator:
    """Stage 6 运行时编排器"""
    
    def __init__(self, config: Stage6Config = None):
        self.config = config or Stage6Config()
        
        # 初始化组件
        self.backbone = BackboneWrapper(self.config)
        self.candidate_generator = CandidateGenerator(self.config)
        self.type_router = TypeRouter(self.config)
        self.param_promoter = None  # 延迟初始化
        self.kb_promoter = KBPromoter(self.config)
        self.rollback_hook = RollbackHook(self.config)
        
        # 指标收集
        self.metrics_history: List[StepMetrics] = []
    
    def run_single_step(self, input_context: str) -> PromotionResult:
        """运行单步晋升"""
        print(f"\n{'='*60}")
        print(f"Stage 6 - 单步晋升")
        print(f"{'='*60}")
        
        try:
            # 1. 主干推理
            print("\n[1/5] 主干推理...")
            input_ids = torch.randint(0, 10000, (1, 50))
            backbone_output = self.backbone.forward(input_ids)
            decisions = backbone_output.get_decisions()
            print(f"  Gap: {decisions['gap']}, Policy: {decisions['policy']}")
            
            # 2. 候选生成
            print("\n[2/5] 候选生成...")
            candidates = self.candidate_generator.generate(input_context, backbone_output)
            print(f"  生成 {len(candidates)} 个候选")
            
            if not candidates:
                return PromotionResult(
                    success=False,
                    target_improvement=0,
                    old_ability_drop=0,
                    writeback_change=0,
                    steps=[],
                    final_kl_weights=self.config.base_kl_weights,
                    error_message="No candidates generated"
                )
            
            # 3. 类型分流
            print("\n[3/5] 类型分流...")
            candidate = candidates[0]  # 取第一个
            route = self.type_router.route(candidate)
            print(f"  候选 {candidate.candidate_id} -> {route}")
            
            if route != 'PARAM_PROMOTION':
                # 知识库晋升
                if route == 'KB_PROMOTION':
                    result = self.kb_promoter.promote(candidate)
                    print(f"  已存入知识库: {result['kb_id']}")
                return PromotionResult(
                    success=True,
                    target_improvement=0,
                    old_ability_drop=0,
                    writeback_change=0,
                    steps=[],
                    final_kl_weights=self.config.base_kl_weights,
                    error_message=f"Routed to {route}, not PARAM_PROMOTION"
                )
            
            # 4. 参数晋升
            print("\n[4/5] 参数晋升...")
            model = self.backbone.get_model()
            
            # 保存基线
            baseline_abilities = self.param_promoter._evaluate_abilities() if self.param_promoter else \
                {'target': 0.5, 'retrieval': 0.5, 'governance': 0.5, 'policy': 0.5, 'writeback': 0.5}
            
            if self.param_promoter is None:
                self.param_promoter = ParamPromoter(model, self.config)
            
            self.rollback_hook.save_baseline(model, baseline_abilities)
            
            # 执行晋升
            step_metrics = self.param_promoter.promote(candidate, 1, baseline_abilities)
            self.metrics_history.append(step_metrics)
            
            print(f"  目标提升: {step_metrics.target_improvement:+.2%}")
            print(f"  旧能力掉落: {step_metrics.old_ability_drop:.2%}")
            print(f"  writeback: {step_metrics.writeback_change:+.2%}")
            
            # 5. 回滚检查
            print("\n[5/5] 回滚检查...")
            need_rollback = self.rollback_hook.check_need_rollback({
                'old_ability_drop': step_metrics.old_ability_drop,
                'writeback_change': step_metrics.writeback_change,
            })
            
            rollback_performed = False
            if need_rollback and self.config.auto_rollback:
                print("  触发回滚!")
                rollback_performed = self.rollback_hook.rollback(model)
                step_metrics.rollback_triggered = True
            else:
                print("  无需回滚")
            
            return PromotionResult(
                success=True,
                target_improvement=step_metrics.target_improvement,
                old_ability_drop=step_metrics.old_ability_drop,
                writeback_change=step_metrics.writeback_change,
                steps=[step_metrics],
                final_kl_weights=self.config.base_kl_weights,
                rollback_performed=rollback_performed,
            )
            
        except Exception as e:
            return PromotionResult(
                success=False,
                target_improvement=0,
                old_ability_drop=0,
                writeback_change=0,
                steps=[],
                final_kl_weights=self.config.base_kl_weights,
                error_message=str(e)
            )
    
    def run_continuous_promotion(self, input_contexts: List[str]) -> List[PromotionResult]:
        """运行连续多步晋升"""
        print(f"\n{'='*60}")
        print(f"Stage 6 - 连续晋升 ({len(input_contexts)} 步)")
        print(f"{'='*60}")
        
        results = []
        for i, context in enumerate(input_contexts):
            print(f"\n--- 第 {i+1}/{len(input_contexts)} 步 ---")
            result = self.run_single_step(context)
            results.append(result)
            
            if not result.success:
                print(f"  第 {i+1} 步失败，停止")
                break
        
        return results
    
    def get_metrics_report(self) -> Dict:
        """获取指标报告"""
        if not self.metrics_history:
            return {'message': 'No metrics collected'}
        
        total_steps = len(self.metrics_history)
        avg_target = sum(m.target_improvement for m in self.metrics_history) / total_steps
        max_old_drop = max(m.old_ability_drop for m in self.metrics_history)
        avg_writeback = sum(m.writeback_change for m in self.metrics_history) / total_steps
        rollback_count = sum(1 for m in self.metrics_history if m.rollback_triggered)
        
        return {
            'total_steps': total_steps,
            'avg_target_improvement': avg_target,
            'max_old_ability_drop': max_old_drop,
            'avg_writeback_change': avg_writeback,
            'rollback_count': rollback_count,
            'step_details': [m.to_dict() for m in self.metrics_history],
        }
    
    def export_report(self, filepath: str):
        """导出报告"""
        report = self.get_metrics_report()
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n报告已导出: {filepath}")


# ==================== 主函数 ====================

def main():
    """主函数 - 演示"""
    print("="*70)
    print("Stage 6 Runtime Orchestrator - 演示")
    print("="*70)
    
    # 创建编排器
    config = Stage6Config()
    orchestrator = Stage6Orchestrator(config)
    
    # 单步测试
    print("\n\n>>> 单步晋升测试 <<<")
    result = orchestrator.run_single_step("测试查询: AI 和机器学习的关系")
    
    if result.success:
        print(f"\n✓ 单步晋升成功")
        print(f"  目标提升: {result.target_improvement:+.2%}")
        print(f"  旧能力掉落: {result.old_ability_drop:.2%}")
        print(f"  writeback: {result.writeback_change:+.2%}")
        if result.rollback_performed:
            print(f"  ⚠️ 已执行回滚")
    else:
        print(f"\n✗ 单步晋升失败: {result.error_message}")
    
    # 连续两步测试
    print("\n\n>>> 连续两步晋升测试 <<<")
    contexts = [
        "查询 1: 深度学习框架对比",
        "查询 2: 神经网络优化算法",
    ]
    results = orchestrator.run_continuous_promotion(contexts)
    
    # 导出报告
    orchestrator.export_report("eval/stage6_orchestrator_report.json")
    
    print("\n" + "="*70)
    print("Stage 6 编排器演示完成")
    print("="*70)


if __name__ == "__main__":
    main()
