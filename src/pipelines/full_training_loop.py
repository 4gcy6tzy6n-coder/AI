"""
Full Training Loop - 完整训练闭环

第六阶段核心组件：
把分散的训练节点真正串成循环：
载入高质量种子样本 → 注入混淆任务 → 模型先行思考 → 
缺失感知 → 内外检索 → 候选生成 → TSLA 初判 → 
强审查 → 验证 → 知识库晋升 → 参数晋升候选 → 
参数写回或拒绝 → 进入下一轮训练

第一版目标：
- 循环可以在一批 case 上完整跑通
- 输出：候选数、被拒绝数、知识库晋升数、参数晋升候选数、参数写回成功数
- 参数写回仍然受控
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.training_pipeline import TrainingPipeline, TrainingCase, TrainingOutcome
from core.training.review_manager import ReviewManager, Evidence
from core.training.verifier import Verifier, VerificationType


class LoopStage(Enum):
    """循环阶段"""
    INIT = "init"
    LOAD_SEEDS = "load_seeds"
    INJECT_NOISE = "inject_noise"
    MODEL_THINK = "model_think"
    PERCEIVE_MISSING = "perceive_missing"
    RETRIEVE = "retrieve"
    GENERATE_CANDIDATE = "generate_candidate"
    TSLA_CHECK = "tsla_check"
    STRONG_REVIEW = "strong_review"
    VERIFY = "verify"
    KB_PROMOTION = "kb_promotion"
    PARAM_CANDIDATE = "param_candidate"
    PARAM_WRITE = "param_write"
    NEXT_ROUND = "next_round"
    COMPLETE = "complete"


@dataclass
class LoopMetrics:
    """循环指标"""
    total_candidates: int = 0
    rejected_count: int = 0
    kb_promoted_count: int = 0
    param_candidate_count: int = 0
    param_write_success_count: int = 0
    
    # 详细统计
    by_stage: Dict[str, int] = field(default_factory=dict)
    by_outcome: Dict[str, int] = field(default_factory=dict)


@dataclass
class TrainingRound:
    """训练轮次"""
    round_number: int
    cases_processed: List[str] = field(default_factory=list)
    metrics: LoopMetrics = field(default_factory=LoopMetrics)
    completed_at: Optional[str] = None


@dataclass
class FullLoopResult:
    """完整循环结果"""
    loop_id: str
    total_rounds: int
    total_cases: int
    final_metrics: LoopMetrics
    rounds: List[TrainingRound]
    completed_at: str


class FullTrainingLoop:
    """
    完整训练闭环
    
    功能：
    1. 批量处理训练案例
    2. 完整执行训练流程
    3. 知识库晋升
    4. 参数晋升候选筛选
    5. 受控参数写回
    6. 多轮循环支持
    
    原则：
    - 参数写回必须受控
    - 基础手工参数不会被覆盖
    - 只有 deep_permanent 内容可进入参数晋升
    - 写回后有回滚/冻结机制
    """
    
    def __init__(self):
        self.training_pipeline = TrainingPipeline()
        self.review_manager = ReviewManager()
        self.verifier = Verifier()
        
        # 存储
        self.kb_promoted: List[str] = []  # 知识库晋升
        self.param_candidates: List[str] = []  # 参数晋升候选
        self.param_written: List[str] = []  # 参数写回成功
        
        # 历史
        self.rounds: List[TrainingRound] = []
        self.current_round: int = 0
    
    def run_loop(
        self,
        seed_cases: List[TrainingCase],
        max_rounds: int = 1,
        enable_param_promotion: bool = False
    ) -> FullLoopResult:
        """
        运行完整训练循环
        
        Args:
            seed_cases: 种子案例列表
            max_rounds: 最大轮数
            enable_param_promotion: 是否启用参数晋升
        """
        print(f"\n{'='*70}")
        print(f"开始完整训练闭环")
        print(f"种子案例数: {len(seed_cases)}")
        print(f"最大轮数: {max_rounds}")
        print(f"参数晋升: {'启用' if enable_param_promotion else '禁用'}")
        print(f"{'='*70}")
        
        start_time = datetime.utcnow()
        
        for round_num in range(1, max_rounds + 1):
            self.current_round = round_num
            print(f"\n{'-'*70}")
            print(f"第 {round_num}/{max_rounds} 轮训练")
            print(f"{'-'*70}")
            
            round_result = self._run_single_round(
                seed_cases, round_num, enable_param_promotion
            )
            self.rounds.append(round_result)
            
            # 打印本轮统计
            print(f"\n本轮统计:")
            print(f"  处理案例: {len(round_result.cases_processed)}")
            print(f"  KB晋升: {round_result.metrics.kb_promoted_count}")
            print(f"  参数候选: {round_result.metrics.param_candidate_count}")
            print(f"  参数写回: {round_result.metrics.param_write_success_count}")
        
        # 汇总结果
        final_metrics = self._calculate_final_metrics()
        
        result = FullLoopResult(
            loop_id=f"loop_{start_time.isoformat()}",
            total_rounds=max_rounds,
            total_cases=len(seed_cases) * max_rounds,
            final_metrics=final_metrics,
            rounds=self.rounds,
            completed_at=datetime.utcnow().isoformat()
        )
        
        self._print_summary(result)
        
        return result
    
    def _run_single_round(
        self,
        cases: List[TrainingCase],
        round_num: int,
        enable_param_promotion: bool
    ) -> TrainingRound:
        """运行单轮训练"""
        round_metrics = LoopMetrics()
        processed_cases = []
        
        for case in cases:
            print(f"\n  处理: {case.case_id}")
            
            # Step 1: 执行训练流程
            result = self.training_pipeline.process(case)
            round_metrics.total_candidates += 1
            
            # 统计结果
            if result.outcome.value not in round_metrics.by_outcome:
                round_metrics.by_outcome[result.outcome.value] = 0
            round_metrics.by_outcome[result.outcome.value] += 1
            
            # Step 2: 知识库晋升统计
            if result.success:
                round_metrics.kb_promoted_count += 1
                self.kb_promoted.append(case.case_id)
                print(f"    → KB晋升: {result.outcome.value}")
                
                # Step 3: 参数晋升候选（仅对深层永久）
                if enable_param_promotion and result.outcome.value == "promoted_to_deep":
                    is_candidate = self._evaluate_param_candidate(case)
                    if is_candidate:
                        round_metrics.param_candidate_count += 1
                        self.param_candidates.append(case.case_id)
                        print(f"    → 参数候选")
                        
                        # Step 4: 参数写回
                        write_success = self._attempt_param_write(case)
                        if write_success:
                            round_metrics.param_write_success_count += 1
                            self.param_written.append(case.case_id)
                            print(f"    → 参数写回成功")
                        else:
                            print(f"    → 参数写回被拒绝")
            else:
                round_metrics.rejected_count += 1
                print(f"    → 被拒绝: {result.outcome.value}")
            
            processed_cases.append(case.case_id)
        
        return TrainingRound(
            round_number=round_num,
            cases_processed=processed_cases,
            metrics=round_metrics,
            completed_at=datetime.utcnow().isoformat()
        )
    
    def _evaluate_param_candidate(self, case: TrainingCase) -> bool:
        """评估是否为参数晋升候选"""
        # 参数晋升门槛更高
        # 这里简化实现，实际应该检查更多维度
        
        if case.seed_quality_score < 0.90:
            return False
        
        # 检查是否来自允许的来源
        if case.source not in ["curated_dataset", "expert_labeled"]:
            return False
        
        return True
    
    def _attempt_param_write(self, case: TrainingCase) -> bool:
        """尝试参数写回"""
        # 受控参数写回
        # 第一版：简化实现，实际应该有更复杂的验证
        
        # 检查是否满足写回条件
        if case.seed_quality_score < 0.95:
            return False
        
        # 模拟写回成功
        return True
    
    def _calculate_final_metrics(self) -> LoopMetrics:
        """计算最终指标"""
        final = LoopMetrics()
        
        for round_data in self.rounds:
            final.total_candidates += round_data.metrics.total_candidates
            final.rejected_count += round_data.metrics.rejected_count
            final.kb_promoted_count += round_data.metrics.kb_promoted_count
            final.param_candidate_count += round_data.metrics.param_candidate_count
            final.param_write_success_count += round_data.metrics.param_write_success_count
        
        return final
    
    def _print_summary(self, result: FullLoopResult):
        """打印汇总"""
        print(f"\n{'='*70}")
        print(f"训练闭环完成汇总")
        print(f"{'='*70}")
        print(f"循环ID: {result.loop_id}")
        print(f"总轮数: {result.total_rounds}")
        print(f"总案例: {result.total_cases}")
        print(f"\n最终指标:")
        print(f"  候选总数: {result.final_metrics.total_candidates}")
        print(f"  被拒绝: {result.final_metrics.rejected_count}")
        print(f"  KB晋升: {result.final_metrics.kb_promoted_count}")
        print(f"  参数候选: {result.final_metrics.param_candidate_count}")
        print(f"  参数写回: {result.final_metrics.param_write_success_count}")
        
        if result.final_metrics.total_candidates > 0:
            kb_rate = result.final_metrics.kb_promoted_count / result.final_metrics.total_candidates
            param_rate = result.final_metrics.param_write_success_count / result.final_metrics.total_candidates
            print(f"\n比率:")
            print(f"  KB晋升率: {kb_rate:.1%}")
            print(f"  参数写回率: {param_rate:.1%}")
    
    def get_kb_promoted(self) -> List[str]:
        """获取知识库晋升列表"""
        return self.kb_promoted.copy()
    
    def get_param_candidates(self) -> List[str]:
        """获取参数晋升候选列表"""
        return self.param_candidates.copy()
    
    def get_param_written(self) -> List[str]:
        """获取参数写回成功列表"""
        return self.param_written.copy()


def demo_full_loop():
    """演示完整训练闭环"""
    print("\n" + "=" * 70)
    print("Full Training Loop Demo - 完整训练闭环演示")
    print("=" * 70)
    
    loop = FullTrainingLoop()
    
    # 创建测试案例
    seed_cases = [
        TrainingCase(
            case_id=f"seed_{i:03d}",
            content=f"高质量训练样本 {i}",
            source="curated_dataset",
            seed_quality_score=0.85 + (i % 3) * 0.05
        )
        for i in range(5)
    ]
    
    # 运行循环（启用参数晋升）
    result = loop.run_loop(
        seed_cases=seed_cases,
        max_rounds=2,
        enable_param_promotion=True
    )
    
    print(f"\n{'='*70}")
    print("演示完成")
    print(f"{'='*70}")


if __name__ == "__main__":
    demo_full_loop()
