"""
Training Pipeline - 训练路径流水线

第五阶段核心组件：
把训练主范式真正串起来：
高质量种子输入 → 长期受审区候选 → 混淆任务注入 → 模型先行思考 → 
缺失感知 → 内部/外部检索 → 候选结论 → TSLA 初判 → 
强审查 → 验证 → 门控晋升 / 回流 / 隔离

第一版目标：训练 case 能进入治理链
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.unit.models import Unit
from core.gates.stability_gate import StabilityGate
from core.gates.promotion_gate import PromotionGate


class TrainingStage(Enum):
    """训练阶段"""
    SEED_INPUT = "seed_input"              # 高质量种子输入
    REVIEW_CANDIDATE = "review_candidate"  # 长期受审区候选
    NOISE_INJECTION = "noise_injection"    # 混淆任务注入
    MODEL_THINKING = "model_thinking"      # 模型先行思考
    MISSING_PERCEPTION = "missing_perception"  # 缺失感知
    RETRIEVAL = "retrieval"                # 内部/外部检索
    CANDIDATE_CONCLUSION = "candidate_conclusion"  # 候选结论
    TSLA_INITIAL = "tsla_initial"          # TSLA 初判
    STRONG_REVIEW = "strong_review"        # 强审查
    VERIFICATION = "verification"          # 验证
    GATE_PROMOTION = "gate_promotion"      # 门控晋升
    REFLOW = "reflow"                      # 回流
    ISOLATION = "isolation"                # 隔离


class TrainingOutcome(Enum):
    """训练结果"""
    PROMOTED_TO_NORMAL = "promoted_to_normal"      # 晋升到正常区
    PROMOTED_TO_DEEP = "promoted_to_deep"          # 晋升到深层永久
    BACK_TO_REVIEW = "back_to_review"              # 回流到受审区
    ISOLATED = "isolated"                          # 隔离
    EXCLUDED = "excluded"                          # 排除
    PENDING_VERIFICATION = "pending_verification"  # 待验证


@dataclass
class TrainingCase:
    """训练案例"""
    case_id: str
    content: str
    source: str                          # 来源类型
    source_details: Dict[str, Any] = field(default_factory=dict)
    seed_quality_score: float = 0.0      # 种子质量分
    stage: TrainingStage = TrainingStage.SEED_INPUT
    outcome: Optional[TrainingOutcome] = None
    
    # 中间结果
    noise_injected: bool = False
    missing_perceived: bool = False
    retrieval_performed: bool = False
    tsla_scores: Dict[str, float] = field(default_factory=dict)
    
    # 审查结果
    review_passed: bool = False
    review_notes: List[str] = field(default_factory=list)
    
    # 验证结果
    verification_passed: bool = False
    verification_rounds: int = 0
    
    # 时间戳
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: Optional[str] = None


@dataclass
class PipelineResult:
    """流水线结果"""
    case_id: str
    success: bool
    final_stage: TrainingStage
    outcome: TrainingOutcome
    duration_ms: int
    logs: List[str] = field(default_factory=list)


class TrainingPipeline:
    """
    训练路径流水线
    
    功能：
    1. 接收高质量种子输入
    2. 执行完整训练流程
    3. 与现有治理系统对接
    4. 输出到长期层或深层永久
    
    注意：第一版只做知识库晋升链，不做参数写回
    """
    
    def __init__(self):
        self.stability_gate = StabilityGate()
        self.promotion_gate = PromotionGate()
        self.cases: Dict[str, TrainingCase] = {}
        
    def process(self, case: TrainingCase) -> PipelineResult:
        """处理单个训练案例"""
        start_time = datetime.utcnow()
        logs = []
        
        def log(msg: str):
            logs.append(f"[{datetime.utcnow().isoformat()}] {msg}")
        
        log(f"开始处理训练案例: {case.case_id}")
        
        try:
            # Stage 1: 种子输入检查
            case = self._check_seed_input(case, log)
            if case.outcome == TrainingOutcome.EXCLUDED:
                return self._create_result(case, logs, start_time)
            
            # Stage 2: 进入受审区候选
            case = self._enter_review_candidate(case, log)
            
            # Stage 3: 混淆任务注入
            case = self._inject_noise(case, log)
            
            # Stage 4: 模型先行思考
            case = self._model_thinking(case, log)
            
            # Stage 5: 缺失感知
            case = self._perceive_missing(case, log)
            
            # Stage 6: 检索
            case = self._perform_retrieval(case, log)
            
            # Stage 7: 生成候选结论
            case = self._generate_conclusion(case, log)
            
            # Stage 8: TSLA 初判
            case = self._tsla_initial_check(case, log)
            if case.outcome in [TrainingOutcome.ISOLATED, TrainingOutcome.EXCLUDED]:
                return self._create_result(case, logs, start_time)
            
            # Stage 9: 强审查
            case = self._strong_review(case, log)
            if not case.review_passed:
                case.outcome = TrainingOutcome.BACK_TO_REVIEW
                return self._create_result(case, logs, start_time)
            
            # Stage 10: 验证
            case = self._verification(case, log)
            if not case.verification_passed:
                case.outcome = TrainingOutcome.PENDING_VERIFICATION
                return self._create_result(case, logs, start_time)
            
            # Stage 11: 门控晋升
            case = self._gate_promotion(case, log)
            
        except Exception as e:
            log(f"处理异常: {str(e)}")
            case.outcome = TrainingOutcome.EXCLUDED
        
        return self._create_result(case, logs, start_time)
    
    def _check_seed_input(self, case: TrainingCase, log) -> TrainingCase:
        """检查种子输入质量"""
        case.stage = TrainingStage.SEED_INPUT
        log(f"Stage 1: 检查种子输入 - 来源: {case.source}")
        
        # 检查来源是否允许
        allowed_sources = ["curated_dataset", "verified_external", "expert_labeled"]
        if case.source not in allowed_sources:
            log(f"  来源 {case.source} 不允许，排除")
            case.outcome = TrainingOutcome.EXCLUDED
            return case
        
        # 检查种子质量
        if case.seed_quality_score < 0.7:
            log(f"  种子质量 {case.seed_quality_score} < 0.7，排除")
            case.outcome = TrainingOutcome.EXCLUDED
            return case
        
        log(f"  种子检查通过，质量分: {case.seed_quality_score}")
        return case
    
    def _enter_review_candidate(self, case: TrainingCase, log) -> TrainingCase:
        """进入受审区候选"""
        case.stage = TrainingStage.REVIEW_CANDIDATE
        log(f"Stage 2: 进入长期受审区候选")
        
        # 创建 Unit 进入 review 区
        unit = Unit(
            unit_id=case.case_id,
            content=case.content,
            source=case.source,
            current_zone="review"
        )
        
        log(f"  已创建 Unit，进入 review 区")
        return case
    
    def _inject_noise(self, case: TrainingCase, log) -> TrainingCase:
        """混淆任务注入"""
        case.stage = TrainingStage.NOISE_INJECTION
        log(f"Stage 3: 混淆任务注入")
        
        # 模拟混淆注入
        case.noise_injected = True
        log(f"  已注入混淆任务，触发模型思考")
        return case
    
    def _model_thinking(self, case: TrainingCase, log) -> TrainingCase:
        """模型先行思考"""
        case.stage = TrainingStage.MODEL_THINKING
        log(f"Stage 4: 模型先行思考")
        
        # 模拟模型思考过程
        log(f"  模型正在处理内容...")
        return case
    
    def _perceive_missing(self, case: TrainingCase, log) -> TrainingCase:
        """缺失感知"""
        case.stage = TrainingStage.MISSING_PERCEPTION
        log(f"Stage 5: 缺失感知")
        
        # 检测是否需要更多信息
        case.missing_perceived = True
        log(f"  检测到信息缺口，需要检索")
        return case
    
    def _perform_retrieval(self, case: TrainingCase, log) -> TrainingCase:
        """执行检索"""
        case.stage = TrainingStage.RETRIEVAL
        log(f"Stage 6: 内部/外部检索")
        
        # 模拟检索
        case.retrieval_performed = True
        log(f"  检索完成，获取支持证据")
        return case
    
    def _generate_conclusion(self, case: TrainingCase, log) -> TrainingCase:
        """生成候选结论"""
        case.stage = TrainingStage.CANDIDATE_CONCLUSION
        log(f"Stage 7: 生成候选结论")
        
        log(f"  候选结论已生成")
        return case
    
    def _tsla_initial_check(self, case: TrainingCase, log) -> TrainingCase:
        """TSLA 初判"""
        case.stage = TrainingStage.TSLA_INITIAL
        log(f"Stage 8: TSLA 初判")
        
        # 模拟 TSLA 评分
        case.tsla_scores = {
            "Q": 0.75 + case.seed_quality_score * 0.2,
            "T": 0.70 + case.seed_quality_score * 0.2,
            "S": 0.65 + case.seed_quality_score * 0.2,
            "E": 0.70 + case.seed_quality_score * 0.15,
            "C": 0.75 + case.seed_quality_score * 0.15,
            "L": 0.80 + case.seed_quality_score * 0.15,
            "R": 0.70 + case.seed_quality_score * 0.1,
            "P": 0.75 + case.seed_quality_score * 0.1
        }
        
        # 检查是否通过初判
        min_score = min(case.tsla_scores.values())
        if min_score < 0.6:
            log(f"  TSLA 初判失败，最低分 {min_score:.2f} < 0.6，隔离")
            case.outcome = TrainingOutcome.ISOLATED
            return case
        
        log(f"  TSLA 初判通过，最低分 {min_score:.2f}")
        return case
    
    def _strong_review(self, case: TrainingCase, log) -> TrainingCase:
        """强审查"""
        case.stage = TrainingStage.STRONG_REVIEW
        log(f"Stage 9: 强审查")
        
        # 来源审查
        if case.source not in ["curated_dataset", "verified_external", "expert_labeled"]:
            log(f"  来源审查未通过")
            case.review_notes.append("来源不符合强审查要求")
            return case
        
        # 证据审查（演示模式下简化）
        if case.seed_quality_score < 0.8 and not case.retrieval_performed:
            log(f"  证据审查未通过，未执行检索")
            case.review_notes.append("缺少检索证据")
            return case
        
        case.review_passed = True
        log(f"  强审查通过")
        return case
    
    def _verification(self, case: TrainingCase, log) -> TrainingCase:
        """验证"""
        case.stage = TrainingStage.VERIFICATION
        log(f"Stage 10: 多轮验证")
        
        # 模拟验证轮数
        case.verification_rounds = 3
        
        # 检查一致性
        if case.tsla_scores["C"] < 0.75:
            log(f"  一致性验证未通过")
            return case
        
        case.verification_passed = True
        log(f"  验证通过，完成 {case.verification_rounds} 轮验证")
        return case
    
    def _gate_promotion(self, case: TrainingCase, log) -> TrainingCase:
        """门控晋升"""
        case.stage = TrainingStage.GATE_PROMOTION
        log(f"Stage 11: 门控晋升")
        
        # 评估是否满足晋升条件
        avg_score = sum(case.tsla_scores.values()) / len(case.tsla_scores)
        
        if avg_score >= 0.85 and case.verification_rounds >= 5:
            # 深层永久候选
            log(f"  满足深层永久条件，avg={avg_score:.2f}")
            case.outcome = TrainingOutcome.PROMOTED_TO_DEEP
        elif avg_score >= 0.75:
            # 长期正常区
            log(f"  晋升到长期正常区，avg={avg_score:.2f}")
            case.outcome = TrainingOutcome.PROMOTED_TO_NORMAL
        else:
            # 回流
            log(f"  回流到受审区，avg={avg_score:.2f}")
            case.outcome = TrainingOutcome.BACK_TO_REVIEW
        
        return case
    
    def _create_result(self, case: TrainingCase, logs: List[str], start_time) -> PipelineResult:
        """创建结果"""
        end_time = datetime.utcnow()
        duration_ms = int((end_time - start_time).total_seconds() * 1000)
        
        case.completed_at = end_time.isoformat()
        self.cases[case.case_id] = case
        
        logs.append(f"[{end_time.isoformat()}] 处理完成: {case.outcome.value if case.outcome else 'unknown'}")
        
        return PipelineResult(
            case_id=case.case_id,
            success=case.outcome in [
                TrainingOutcome.PROMOTED_TO_NORMAL,
                TrainingOutcome.PROMOTED_TO_DEEP
            ],
            final_stage=case.stage,
            outcome=case.outcome or TrainingOutcome.EXCLUDED,
            duration_ms=duration_ms,
            logs=logs
        )
    
    def get_case(self, case_id: str) -> Optional[TrainingCase]:
        """获取案例"""
        return self.cases.get(case_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total = len(self.cases)
        outcomes = {}
        for case in self.cases.values():
            outcome = case.outcome.value if case.outcome else "unknown"
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
        
        return {
            "total_cases": total,
            "outcomes": outcomes,
            "success_rate": outcomes.get("promoted_to_normal", 0) / total if total > 0 else 0
        }


def run_training_demo():
    """运行训练演示"""
    print("\n" + "=" * 70)
    print("Training Pipeline Demo - 训练路径演示")
    print("=" * 70)
    
    pipeline = TrainingPipeline()
    
    # 创建测试案例
    test_cases = [
        TrainingCase(
            case_id="train_001",
            content="高质量训练样本A",
            source="curated_dataset",
            seed_quality_score=0.9
        ),
        TrainingCase(
            case_id="train_002",
            content="高质量训练样本B",
            source="verified_external",
            seed_quality_score=0.85
        ),
        TrainingCase(
            case_id="train_003",
            content="低质量训练样本",
            source="unverified",
            seed_quality_score=0.5
        )
    ]
    
    for case in test_cases:
        print(f"\n处理案例: {case.case_id}")
        print("-" * 50)
        result = pipeline.process(case)
        
        print(f"  结果: {result.outcome.value}")
        print(f"  耗时: {result.duration_ms}ms")
        print(f"  成功: {'✅' if result.success else '❌'}")
    
    # 打印统计
    print("\n" + "=" * 70)
    print("统计信息")
    print("=" * 70)
    stats = pipeline.get_stats()
    print(f"  总案例数: {stats['total_cases']}")
    print(f"  结果分布: {stats['outcomes']}")
    print(f"  成功率: {stats['success_rate']:.1%}")


if __name__ == "__main__":
    run_training_demo()
