"""
Stage 6 System Orchestrator - Phase 3 Official Entry

Phase 3 统一官方系统壳

职责:
1. 系统总入口 - 配置加载、模型加载、评估器挂载
2. 系统级运行拓扑 - 最小闭环 (输入→检索→推理→writeback→rollback→评估→落盘)
3. 环境一致性规范 - baseline版本、checkpoint命名、随机种子
4. 监控与追踪 - 指标收集、step-level trace、异常日志
5. 预留接口 - Task 2/3 入口

官方基线: stage6_official_baseline_v1.md
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import json
import copy
import random
import numpy as np
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum

from stage6_runtime_orchestrator import Stage6Orchestrator, Stage6Config
from stage6_evaluation_protocol import UnifiedEvaluationProtocol
from stage6_full_rollback_snapshot import FullRollbackManager


# ==================== 官方基线配置 ====================

OFFICIAL_BASELINE_V1 = {
    'version': '1.0',
    'base_kl_weights': {
        'gap': 0.30,
        'policy': 0.30,
        'governance': 0.30,
        'writeback': 0.42,
    },
    'learning_rate': 1.0e-5,
    'replay_ratio': 0.40,
    'step1_max_change': 0.003,
    'step2_max_change': 0.008,
    'old_ability_threshold': 0.12,
    'writeback_threshold': 0.05,
    'auto_rollback': True,
    'param_promotion_threshold': 0.80,
    'kb_promotion_threshold': 0.70,
}


# ==================== 环境一致性规范 ====================

class EnvironmentSpec:
    """环境一致性规范"""
    
    # 唯一 baseline 版本
    BASELINE_VERSION = '1.0'
    
    # 唯一 checkpoint 命名格式
    CHECKPOINT_FORMAT = '{exp_id}_{timestamp}_{step}'
    
    # 唯一数据切分 (固定种子)
    DATA_SPLIT_SEED = 42
    
    # 唯一随机种子策略
    TORCH_SEED = 42
    NUMPY_SEED = 42
    RANDOM_SEED = 42
    
    # 唯一指标输出格式
    METRICS_FORMAT = 'json'
    
    @classmethod
    def set_global_seeds(cls):
        """设置全局随机种子"""
        torch.manual_seed(cls.TORCH_SEED)
        np.random.seed(cls.NUMPY_SEED)
        random.seed(cls.RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.TORCH_SEED)
    
    @classmethod
    def generate_checkpoint_id(cls, exp_id: str, step: int) -> str:
        """生成标准 checkpoint ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return cls.CHECKPOINT_FORMAT.format(
            exp_id=exp_id,
            timestamp=timestamp,
            step=step
        )


# ==================== 系统状态 ====================

class SystemState(Enum):
    """系统状态"""
    INITIALIZED = 'initialized'
    BASELINE_ESTABLISHED = 'baseline_established'
    RUNNING = 'running'
    EVALUATING = 'evaluating'
    ROLLBACK = 'rollback'
    COMPLETED = 'completed'
    ERROR = 'error'


@dataclass
class StepTrace:
    """单步追踪记录"""
    step_number: int
    timestamp: str
    state: str
    
    # 输入
    input_query: str = ''
    
    # 推理结果
    gap_detected: int = -1
    policy_selected: int = -1
    
    # 晋升结果
    promotion_performed: bool = False
    promotion_type: str = ''
    
    # 评估指标
    target_improvement: float = 0.0
    old_ability_drop: float = 0.0
    writeback_change: float = 0.0
    
    # 回滚
    rollback_triggered: bool = False
    rollback_success: bool = False
    
    # 异常
    error_message: str = ''


@dataclass
class SystemMetrics:
    """系统级指标"""
    experiment_id: str
    start_time: str
    end_time: Optional[str] = None
    
    # 累积指标
    total_steps: int = 0
    total_promotions: int = 0
    total_rollbacks: int = 0
    
    # 最终指标
    final_target_gain: float = 0.0
    max_old_ability_drop: float = 0.0
    max_writeback_change: float = 0.0
    rollback_recovery_rate: float = 0.0
    
    # 追踪历史
    step_traces: List[StepTrace] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'experiment_id': self.experiment_id,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'total_steps': self.total_steps,
            'total_promotions': self.total_promotions,
            'total_rollbacks': self.total_rollbacks,
            'final_target_gain': self.final_target_gain,
            'max_old_ability_drop': self.max_old_ability_drop,
            'max_writeback_change': self.max_writeback_change,
            'rollback_recovery_rate': self.rollback_recovery_rate,
            'step_traces': [asdict(t) for t in self.step_traces],
        }


# ==================== 系统编排器 ====================

class Stage6SystemOrchestrator:
    """
    Stage 6 Phase 3 统一系统编排器
    
    这是 Phase 3 的唯一官方系统入口
    """
    
    def __init__(self, experiment_id: str = None, custom_config: Dict = None):
        """
        初始化系统编排器
        
        Args:
            experiment_id: 实验ID (自动生成如果未提供)
            custom_config: 自定义配置 (覆盖官方基线)
        """
        # 设置环境一致性
        EnvironmentSpec.set_global_seeds()
        
        # 实验标识
        self.experiment_id = experiment_id or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.start_time = datetime.now().isoformat()
        
        print(f"\n{'='*70}")
        print(f"Stage 6 Phase 3 - 系统编排器")
        print(f"{'='*70}")
        print(f"实验ID: {self.experiment_id}")
        print(f"Baseline版本: {EnvironmentSpec.BASELINE_VERSION}")
        print(f"开始时间: {self.start_time}")
        
        # 加载配置 (官方基线 + 自定义覆盖)
        self.config = copy.deepcopy(OFFICIAL_BASELINE_V1)
        if custom_config:
            self.config.update(custom_config)
        print(f"\n配置加载完成 (版本: {self.config['version']})")
        
        # 初始化核心组件
        self._init_core_components()
        
        # 系统状态
        self.state = SystemState.INITIALIZED
        self.metrics = SystemMetrics(
            experiment_id=self.experiment_id,
            start_time=self.start_time,
        )
        
        # 回调函数 (用于监控)
        self.step_callbacks: List[Callable] = []
        self.error_callbacks: List[Callable] = []
        
        print(f"\n✓ 系统编排器初始化完成")
        print(f"{'='*70}\n")
    
    def _init_core_components(self):
        """初始化核心组件"""
        print("[1/4] 初始化 Stage6 编排器...")
        # 过滤掉非配置项 (如 version, name, description)
        config_for_orchestrator = {k: v for k, v in self.config.items() 
                                   if k not in ['version', 'name', 'description']}
        self.orchestrator = Stage6Orchestrator(Stage6Config(**config_for_orchestrator))
        
        print("[2/4] 获取模型...")
        self.model = self.orchestrator.backbone.get_model()
        
        print("[3/4] 初始化评估协议...")
        self.protocol = UnifiedEvaluationProtocol(self.model)
        
        print("[4/4] 初始化 rollback 管理器...")
        optimizer = self.orchestrator.param_promoter.optimizer if self.orchestrator.param_promoter else None
        self.rollback_manager = FullRollbackManager(self.model, optimizer)
    
    def establish_baseline(self) -> str:
        """
        建立系统基线
        
        Returns:
            checkpoint_id: 基线 checkpoint ID
        """
        print(f"\n{'='*70}")
        print("建立系统基线")
        print(f"{'='*70}")
        
        self.state = SystemState.BASELINE_ESTABLISHED
        
        # 使用统一协议建立基线
        self.protocol.establish_baseline(f"{self.experiment_id}_baseline")
        
        # 创建 rollback 快照
        checkpoint_id = EnvironmentSpec.generate_checkpoint_id(self.experiment_id, 0)
        self.rollback_manager.create_snapshot(checkpoint_id, {
            'description': 'Phase 3 system baseline',
            'baseline_scores': self.protocol.baseline.scores.to_dict(),
        })
        
        print(f"\n✓ 基线已建立: {checkpoint_id}")
        print(f"{'='*70}\n")
        
        return checkpoint_id
    
    def run_system_loop(self, queries: List[str], max_steps: int = None) -> SystemMetrics:
        """
        运行系统级闭环
        
        拓扑: 输入 → 检索/上下文 → 推理 → writeback → rollback → 评估 → 落盘
        
        Args:
            queries: 输入查询列表
            max_steps: 最大步数 (默认全部)
            
        Returns:
            系统指标
        """
        if self.state == SystemState.INITIALIZED:
            self.establish_baseline()
        
        max_steps = max_steps or len(queries)
        
        print(f"\n{'='*70}")
        print(f"运行系统级闭环 (max_steps={max_steps})")
        print(f"{'='*70}\n")
        
        for step_num in range(1, min(max_steps + 1, len(queries) + 1)):
            query = queries[step_num - 1]
            
            # 执行单步
            trace = self._execute_step(step_num, query)
            self.metrics.step_traces.append(trace)
            
            # 触发回调
            for callback in self.step_callbacks:
                callback(step_num, trace)
        
        # 计算最终指标
        self._compute_final_metrics()
        
        self.state = SystemState.COMPLETED
        self.metrics.end_time = datetime.now().isoformat()
        
        return self.metrics
    
    def _execute_step(self, step_num: int, query: str) -> StepTrace:
        """执行单步闭环"""
        trace = StepTrace(
            step_number=step_num,
            timestamp=datetime.now().isoformat(),
            state='running',
            input_query=query,
        )
        
        print(f"\n--- Step {step_num} ---")
        
        try:
            self.state = SystemState.RUNNING
            
            # 1. 检索/上下文装配 (在 orchestrator 中处理)
            # 2. 推理 + writeback + rollback (统一在 orchestrator 中)
            torch.manual_seed(EnvironmentSpec.TORCH_SEED + step_num)
            result = self.orchestrator.run_single_step(query)
            
            trace.gap_detected = result.steps[0].gap if result.steps else -1
            trace.policy_selected = result.steps[0].policy if result.steps else -1
            trace.promotion_performed = len(result.steps) > 0
            trace.promotion_type = 'PARAM' if result.steps else 'NONE'
            trace.rollback_triggered = result.rollback_performed
            
            # 3. 评估
            self.state = SystemState.EVALUATING
            eval_result = self.protocol.evaluate_with_protocol(f"step_{step_num}", num_samples=30)
            
            trace.target_improvement = eval_result.target_improvement
            trace.old_ability_drop = eval_result.old_ability_drop
            trace.writeback_change = eval_result.writeback_change
            
            # 更新累积指标
            self.metrics.total_steps += 1
            if trace.promotion_performed:
                self.metrics.total_promotions += 1
            if trace.rollback_triggered:
                self.metrics.total_rollbacks += 1
            
            print(f"  target: {trace.target_improvement:+.2%}, old_drop: {trace.old_ability_drop:.2%}, writeback: {trace.writeback_change:+.2%}")
            
        except Exception as e:
            self.state = SystemState.ERROR
            trace.state = 'error'
            trace.error_message = str(e)
            print(f"  ✗ 错误: {e}")
            
            # 触发错误回调
            for callback in self.error_callbacks:
                callback(step_num, e)
        
        return trace
    
    def _compute_final_metrics(self):
        """计算最终指标"""
        if not self.metrics.step_traces:
            return
        
        # 从追踪记录计算
        final_trace = self.metrics.step_traces[-1]
        self.metrics.final_target_gain = final_trace.target_improvement
        self.metrics.max_old_ability_drop = max(t.old_ability_drop for t in self.metrics.step_traces)
        self.metrics.max_writeback_change = max(abs(t.writeback_change) for t in self.metrics.step_traces)
        
        # rollback 恢复率 (简化计算)
        if self.metrics.total_rollbacks > 0:
            successful_rollbacks = sum(1 for t in self.metrics.step_traces if t.rollback_triggered and not t.error_message)
            self.metrics.rollback_recovery_rate = (successful_rollbacks / self.metrics.total_rollbacks) * 100
    
    def test_rollback_recovery(self) -> float:
        """测试 rollback 恢复能力"""
        print(f"\n{'='*70}")
        print("测试 Rollback 恢复")
        print(f"{'='*70}")
        
        # 保存当前状态
        current_scores = self.protocol.evaluator.evaluate_all(num_samples=30).scores
        
        # 恢复到基线
        self.state = SystemState.ROLLBACK
        baseline_checkpoint = f"{self.experiment_id}_baseline"
        success = self.rollback_manager.restore_snapshot(baseline_checkpoint)
        
        if not success:
            print("✗ 恢复失败")
            return 0.0
        
        # 评估恢复后状态
        recovered_scores = self.protocol.evaluator.evaluate_all(num_samples=30).scores
        
        # 计算恢复率
        baseline_scores = self.protocol.baseline.scores
        recovery_rates = []
        
        for key in ['target_score', 'retrieval_score', 'policy_score', 'governance_score', 'writeback_score']:
            baseline_val = getattr(baseline_scores, key, 0)
            if baseline_val > 0:
                current_val = getattr(current_scores, key, 0)
                recovered_val = getattr(recovered_scores, key, 0)
                
                pre_deviation = abs(current_val - baseline_val) / baseline_val
                post_deviation = abs(recovered_val - baseline_val) / baseline_val
                
                if pre_deviation > 0:
                    recovery_rate = max(0, (pre_deviation - post_deviation) / pre_deviation * 100)
                    recovery_rates.append(recovery_rate)
        
        avg_recovery = sum(recovery_rates) / len(recovery_rates) if recovery_rates else 0
        self.metrics.rollback_recovery_rate = avg_recovery
        
        print(f"\n✓ 恢复率: {avg_recovery:.1f}%")
        return avg_recovery
    
    def export_report(self, filepath: str = None):
        """导出系统报告"""
        filepath = filepath or f"eval/{self.experiment_id}_report.json"
        
        report = {
            'experiment_id': self.experiment_id,
            'baseline_version': EnvironmentSpec.BASELINE_VERSION,
            'config': self.config,
            'metrics': self.metrics.to_dict(),
            'final_state': self.state.value,
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n✓ 报告已导出: {filepath}")
        return filepath
    
    def register_step_callback(self, callback: Callable):
        """注册步进回调"""
        self.step_callbacks.append(callback)
    
    def register_error_callback(self, callback: Callable):
        """注册错误回调"""
        self.error_callbacks.append(callback)


# ==================== 便捷函数 ====================

def run_official_phase3_test(num_steps: int = 5) -> SystemMetrics:
    """
    运行官方 Phase 3 测试
    
    这是 Phase 3 的标准测试入口
    """
    # 创建系统编排器
    system = Stage6SystemOrchestrator(experiment_id='phase3_official_test')
    
    # 准备测试数据
    queries = [f"查询 {i+1}: 系统级测试查询" for i in range(num_steps)]
    
    # 运行系统闭环
    metrics = system.run_system_loop(queries)
    
    # 测试 rollback
    system.test_rollback_recovery()
    
    # 导出报告
    system.export_report()
    
    # 打印摘要
    print(f"\n{'='*70}")
    print("Phase 3 官方测试完成")
    print(f"{'='*70}")
    print(f"总步数: {metrics.total_steps}")
    print(f"总晋升: {metrics.total_promotions}")
    print(f"总回滚: {metrics.total_rollbacks}")
    print(f"目标提升: {metrics.final_target_gain:+.2%}")
    print(f"最大旧能力掉落: {metrics.max_old_ability_drop:.2%}")
    print(f"最大 writeback 变化: {metrics.max_writeback_change:.2%}")
    print(f"Rollback 恢复率: {metrics.rollback_recovery_rate:.1f}%")
    
    return metrics


# ==================== 测试 ====================

def test_system_orchestrator():
    """测试系统编排器"""
    print("\n" + "="*70)
    print("测试 Stage 6 系统编排器")
    print("="*70)
    
    # 创建编排器
    system = Stage6SystemOrchestrator(experiment_id='test_system')
    
    # 建立基线
    checkpoint_id = system.establish_baseline()
    print(f"基线 checkpoint: {checkpoint_id}")
    
    # 运行 3 步测试
    queries = ["测试查询 1", "测试查询 2", "测试查询 3"]
    metrics = system.run_system_loop(queries)
    
    # 导出报告
    report_path = system.export_report()
    
    print(f"\n✓ 系统编排器测试完成")
    print(f"报告: {report_path}")


if __name__ == "__main__":
    # 运行官方测试
    run_official_phase3_test(num_steps=5)
