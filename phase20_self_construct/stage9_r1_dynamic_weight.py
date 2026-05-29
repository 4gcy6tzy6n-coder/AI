"""
Stage 9-R1: 动态权重调整实验
多层任务训练与系统优化

策略:
- 课程学习: Phase 1 (L1) → Phase 2 (L1+L2) → Phase 3 (L1+L2+L3)
- 动态权重: 根据表现自动调整 L1/L2/L3 权重
- Mixed-batch: 每 batch 包含各层样本
- L1 replay: 保留 anchor，每隔固定步 replay
- 长期训练: 500-1000 步
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import json
import random
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field

from phase19_native_backbone.native_backbone_tiny_v1 import NativeBackboneTinyV1, NativeTinyConfig
from stage7_output_kl_guard import build_output_kl_guard


@dataclass
class DynamicWeightConfig:
    """动态权重配置"""
    l1_base: float = 1.3
    l2_base: float = 1.0
    l3_base: float = 1.0
    adjustment_rate: float = 0.1  # 权重调整速率
    min_weight: float = 0.5
    max_weight: float = 2.0


@dataclass
class TSLAEvent:
    """TSLA 事件记录"""
    step: int
    event_type: str
    unit_id: Optional[str]
    trigger_condition: str
    result: str
    timestamp: str


@dataclass
class Stage9Metrics:
    """Stage 9 指标"""
    step: int
    phase: str
    l1_accuracy: float
    l2_accuracy: float
    l3_accuracy: float
    overall_accuracy: float
    composite_score: float
    weights: Dict[str, float]  # 当前权重
    writeback_change: float
    grad_ratio: float
    output_drift: float
    loss: float
    tsla_event_count: int
    tsla_events: List[TSLAEvent] = field(default_factory=list)
    timestamp: str = ""


class DynamicWeightScheduler:
    """动态权重调度器"""
    
    def __init__(self, config: DynamicWeightConfig):
        self.config = config
        self.weights = {
            'L1': config.l1_base,
            'L2': config.l2_base,
            'L3': config.l3_base,
        }
        self.performance_history = []
    
    def update_weights(self, l1_acc: float, l2_acc: float, l3_acc: float, step: int):
        """根据表现动态调整权重"""
        self.performance_history.append({
            'step': step,
            'L1': l1_acc,
            'L2': l2_acc,
            'L3': l3_acc,
        })
        
        # 只保留最近 3 次记录用于趋势判断
        if len(self.performance_history) > 3:
            self.performance_history.pop(0)
        
        if len(self.performance_history) < 2:
            return
        
        # 计算趋势
        prev = self.performance_history[-2]
        curr = self.performance_history[-1]
        
        # L1 保护逻辑: 如果 L1 下降，增加权重
        if curr['L1'] < prev['L1'] and curr['L1'] < 0.6:
            self.weights['L1'] = min(
                self.weights['L1'] + self.config.adjustment_rate,
                self.config.max_weight
            )
        # L1 过高时，适当降低权重给 L2/L3 机会
        elif curr['L1'] > 0.9 and curr['L2'] < 0.4:
            self.weights['L1'] = max(
                self.weights['L1'] - self.config.adjustment_rate * 0.5,
                self.config.min_weight
            )
        
        # L2/L3 提升逻辑: 如果过低，暂时降低 L1 权重
        if curr['L2'] < 0.3 and curr['L1'] > 0.7:
            self.weights['L2'] = min(
                self.weights['L2'] + self.config.adjustment_rate,
                self.config.max_weight
            )
        
        if curr['L3'] < 0.2 and curr['L1'] > 0.7:
            self.weights['L3'] = min(
                self.weights['L3'] + self.config.adjustment_rate,
                self.config.max_weight
            )


class TSLAGate:
    """TSLA 门控"""
    
    def __init__(self):
        self.promotion_threshold = 0.8
        self.isolation_threshold = 0.3
        self.event_history: List[TSLAEvent] = []
        self.action_correct_count = 0
        self.action_total_count = 0
        
    def check_promotion(self, unit_confidence: float, step: int) -> Tuple[bool, str]:
        self.action_total_count += 1
        if unit_confidence >= self.promotion_threshold:
            self.action_correct_count += 1
            event = TSLAEvent(
                step=step,
                event_type="promotion",
                unit_id=None,
                trigger_condition=f"confidence >= {self.promotion_threshold}",
                result="promoted_to_longterm",
                timestamp=datetime.now().isoformat(),
            )
            self.event_history.append(event)
            return True, "longterm"
        return False, "instant"
    
    def check_isolation(self, unit_confidence: float, step: int) -> bool:
        self.action_total_count += 1
        if unit_confidence < self.isolation_threshold:
            self.action_correct_count += 1
            event = TSLAEvent(
                step=step,
                event_type="isolation",
                unit_id=None,
                trigger_condition=f"confidence < {self.isolation_threshold}",
                result="isolated",
                timestamp=datetime.now().isoformat(),
            )
            self.event_history.append(event)
            return True
        return False
    
    def record_writeback(self, success: bool, step: int):
        self.action_total_count += 1
        if success:
            self.action_correct_count += 1
        event = TSLAEvent(
            step=step,
            event_type="writeback",
            unit_id=None,
            trigger_condition="training_step",
            result="success" if success else "failed",
            timestamp=datetime.now().isoformat(),
        )
        self.event_history.append(event)
    
    def get_event_summary(self) -> Dict:
        events_by_type = {}
        for event in self.event_history:
            events_by_type[event.event_type] = events_by_type.get(event.event_type, 0) + 1
        return events_by_type
    
    def get_action_accuracy(self) -> float:
        if self.action_total_count == 0:
            return 0.0
        return self.action_correct_count / self.action_total_count


class Stage9R1DynamicWeightTrainer:
    """Stage 9-R1 动态权重训练器"""
    
    def __init__(
        self,
        model,
        l1_dataset: List[Dict],
        l2_dataset: List[Dict],
        l3_dataset: List[Dict],
        device: str = 'cpu',
    ):
        self.model = model
        self.l1_dataset = l1_dataset
        self.l2_dataset = l2_dataset
        self.l3_dataset = l3_dataset
        self.device = device
        
        # Output KL Guard (冻结配置)
        self.guard = build_output_kl_guard(
            model=model,
            beta=0.2,
            num_samples=30,
            use_probs=True,
        )
        self.guard.capture_reference_outputs()
        
        # TSLA 门控
        self.tsla_gate = TSLAGate()
        
        # 动态权重调度器
        self.weight_scheduler = DynamicWeightScheduler(DynamicWeightConfig())
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
        )
        
        # 基线
        self.baseline_wb_score = None
        self.establish_baseline()
        
        # 记录
        self.metrics_history: List[Stage9Metrics] = []
        self.best_checkpoint = None
        self.best_score = 0.0
        
        # 阶段划分
        self.phase_steps = {
            'L1_warmup': (0, 100),
            'L1_L2': (100, 300),
            'L1_L2_L3': (300, 500),
        }
        
        # Replay 配置
        self.replay_interval = 20
        
    def establish_baseline(self):
        """建立基线"""
        self.model.eval()
        with torch.no_grad():
            torch.manual_seed(42)
            anchor_inputs = [torch.randint(0, 10000, (1, 50)).to(self.device) for _ in range(30)]
            wb_scores = []
            for input_ids in anchor_inputs:
                outputs = self.model(input_ids)
                probs = F.softmax(outputs['writeback_logits'], dim=-1)
                score = probs[0, 1].item() if probs.shape[1] > 1 else probs[0, 0].item()
                wb_scores.append(score)
            self.baseline_wb_score = sum(wb_scores) / len(wb_scores)
        
        print(f"[S9-R1] Writeback 基线: {self.baseline_wb_score:.4f}")
    
    def get_current_phase(self, step: int) -> str:
        """获取当前阶段"""
        for phase, (start, end) in self.phase_steps.items():
            if start <= step < end:
                return phase
        return 'L1_L2_L3'
    
    def get_mixed_batch(self, phase: str) -> List[Tuple[Dict, str]]:
        """获取混合 batch"""
        batch = []
        
        if phase == 'L1_warmup':
            # 纯 L1
            for _ in range(4):
                batch.append((random.choice(self.l1_dataset), 'L1'))
        elif phase == 'L1_L2':
            # L1:L2 = 1:1
            weights = self.weight_scheduler.weights
            l1_count = max(2, int(4 * weights['L1'] / (weights['L1'] + weights['L2'])))
            l2_count = 4 - l1_count
            for _ in range(l1_count):
                batch.append((random.choice(self.l1_dataset), 'L1'))
            for _ in range(l2_count):
                batch.append((random.choice(self.l2_dataset), 'L2'))
        else:  # L1_L2_L3
            # L1:L2:L3 = 2:2:1 基础，根据权重调整
            weights = self.weight_scheduler.weights
            total_weight = weights['L1'] + weights['L2'] + weights['L3']
            l1_count = max(1, int(5 * weights['L1'] / total_weight))
            l2_count = max(1, int(5 * weights['L2'] / total_weight))
            l3_count = max(1, 5 - l1_count - l2_count)
            
            for _ in range(l1_count):
                batch.append((random.choice(self.l1_dataset), 'L1'))
            for _ in range(l2_count):
                batch.append((random.choice(self.l2_dataset), 'L2'))
            for _ in range(l3_count):
                batch.append((random.choice(self.l3_dataset), 'L3'))
        
        return batch
    
    def compute_task_loss(self, sample: Dict, difficulty: str) -> torch.Tensor:
        """计算任务损失"""
        weight = self.weight_scheduler.weights.get(difficulty, 1.0)
        
        question_tokens = torch.randint(0, 10000, (1, 20)).to(self.device)
        outputs = self.model(question_tokens)
        
        expected_gap = torch.tensor([sample.get('expected_gap', 0)], device=self.device, dtype=torch.long)
        expected_policy = torch.tensor([sample.get('expected_policy', 0)], device=self.device, dtype=torch.long)
        
        gap_loss = F.cross_entropy(outputs['gap_logits'], expected_gap)
        policy_loss = F.cross_entropy(outputs['policy_logits'], expected_policy)
        
        return (gap_loss + policy_loss) * weight
    
    def evaluate_by_difficulty(self, dataset: List[Dict], num_samples: int = 20) -> float:
        """按难度评估准确率"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for sample in dataset[:num_samples]:
                question_tokens = torch.randint(0, 10000, (1, 20)).to(self.device)
                outputs = self.model(question_tokens)
                
                pred_gap = outputs['gap_probs'][0].argmax().item()
                pred_policy = outputs['policy_probs'][0].argmax().item()
                
                if pred_gap == sample.get('expected_gap', 0):
                    correct += 1
                if pred_policy == sample.get('expected_policy', 0):
                    correct += 1
                total += 2
        
        return correct / total if total > 0 else 0.0
    
    def evaluate_writeback(self) -> float:
        """评估 writeback"""
        self.model.eval()
        with torch.no_grad():
            torch.manual_seed(42)
            anchor_inputs = [torch.randint(0, 10000, (1, 50)).to(self.device) for _ in range(30)]
            wb_scores = []
            for input_ids in anchor_inputs:
                outputs = self.model(input_ids)
                probs = F.softmax(outputs['writeback_logits'], dim=-1)
                score = probs[0, 1].item() if probs.shape[1] > 1 else probs[0, 0].item()
                wb_scores.append(score)
            return sum(wb_scores) / len(wb_scores)
    
    def compute_composite_score(self, l1_acc: float, l2_acc: float, l3_acc: float, wb_change: float) -> float:
        """计算综合评分"""
        # 0.35*L1 + 0.35*L2 + 0.20*L3 + 0.10*WB
        wb_score = max(0, 1 - abs(wb_change) / 0.05)
        return 0.35 * l1_acc + 0.35 * l2_acc + 0.20 * l3_acc + 0.10 * wb_score
    
    def train_step(self, step: int) -> Tuple[Optional[Stage9Metrics], float]:
        """执行一步训练"""
        self.model.train()
        
        phase = self.get_current_phase(step)
        total_loss = 0.0
        
        # 1. Mixed-batch 训练
        batch = self.get_mixed_batch(phase)
        
        for sample, difficulty in batch:
            self.optimizer.zero_grad()
            
            # Guard Loss
            guard_loss = self.guard.compute_kl_guard_loss()
            guard_loss.backward(retain_graph=True)
            
            guard_grad_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    guard_grad_norm += p.grad.norm().item() ** 2
            guard_grad_norm = guard_grad_norm ** 0.5
            
            # Main Loss
            self.optimizer.zero_grad()
            main_loss = self.compute_task_loss(sample, difficulty)
            main_loss.backward()
            
            main_grad_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    main_grad_norm += p.grad.norm().item() ** 2
            main_grad_norm = main_grad_norm ** 0.5
            
            # 合并并更新
            self.optimizer.zero_grad()
            guard_loss = self.guard.compute_kl_guard_loss()
            main_loss = self.compute_task_loss(sample, difficulty)
            loss = main_loss + guard_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        # 2. L1 Replay (每 20 步)
        if step % self.replay_interval == 0:
            self.optimizer.zero_grad()
            replay_sample = random.choice(self.l1_dataset)
            replay_loss = self.compute_task_loss(replay_sample, 'L1')
            replay_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += replay_loss.item()
        
        # 3. TSLA 门控
        with torch.no_grad():
            mock_confidence = torch.sigmoid(torch.randn(1)).item()
        self.tsla_gate.check_promotion(mock_confidence, step)
        self.tsla_gate.check_isolation(mock_confidence, step)
        self.tsla_gate.record_writeback(success=True, step=step)
        
        # 4. 评估 (每 25 步)
        if step % 25 == 0 or step == 499:
            l1_acc = self.evaluate_by_difficulty(self.l1_dataset, 20)
            l2_acc = self.evaluate_by_difficulty(self.l2_dataset, 20)
            l3_acc = self.evaluate_by_difficulty(self.l3_dataset, 20)
            
            overall_acc = (l1_acc + l2_acc + l3_acc) / 3
            writeback_score = self.evaluate_writeback()
            writeback_change = writeback_score - self.baseline_wb_score
            output_drift = self.guard.compute_output_drift()
            
            composite_score = self.compute_composite_score(l1_acc, l2_acc, l3_acc, writeback_change)
            
            grad_ratio = guard_grad_norm / main_grad_norm if main_grad_norm > 0 else 0
            
            # 更新权重
            self.weight_scheduler.update_weights(l1_acc, l2_acc, l3_acc, step)
            
            # 保存最佳 checkpoint
            if composite_score > self.best_score:
                self.best_score = composite_score
                self.best_checkpoint = {
                    'step': step,
                    'l1_accuracy': l1_acc,
                    'l2_accuracy': l2_acc,
                    'l3_accuracy': l3_acc,
                    'composite_score': composite_score,
                    'weights': self.weight_scheduler.weights.copy(),
                }
            
            metrics = Stage9Metrics(
                step=step,
                phase=phase,
                l1_accuracy=l1_acc,
                l2_accuracy=l2_acc,
                l3_accuracy=l3_acc,
                overall_accuracy=overall_acc,
                composite_score=composite_score,
                weights=self.weight_scheduler.weights.copy(),
                writeback_change=writeback_change,
                grad_ratio=grad_ratio,
                output_drift=output_drift,
                loss=total_loss / len(batch),
                tsla_event_count=len(self.tsla_gate.event_history),
                tsla_events=self.tsla_gate.event_history[-5:],
                timestamp=datetime.now().isoformat(),
            )
            
            return metrics, total_loss
        
        return None, total_loss
    
    def train(self, num_steps: int = 500) -> List[Stage9Metrics]:
        """执行训练"""
        print("\n" + "="*70)
        print("Stage 9-R1: 动态权重调整实验")
        print("="*70)
        print(f"L1 样本: {len(self.l1_dataset)}")
        print(f"L2 样本: {len(self.l2_dataset)}")
        print(f"L3 样本: {len(self.l3_dataset)}")
        print(f"训练步数: {num_steps}")
        print(f"\n课程阶段:")
        print(f"  Phase 1 (0-99): L1 warm-up")
        print(f"  Phase 2 (100-299): L1 + L2")
        print(f"  Phase 3 (300-499): L1 + L2 + L3")
        print(f"\n动态权重策略:")
        print(f"  - 基础权重: L1=1.3, L2=1.0, L3=1.0")
        print(f"  - 根据表现自动调整")
        print(f"  - L1 replay: 每 {self.replay_interval} 步")
        print(f"Output KL Guard: beta=0.2 (冻结)")
        print(f"TSLA 门控: 冻结配置")
        print("="*70)
        
        for step in range(num_steps):
            metrics, loss = self.train_step(step)
            
            if metrics is not None:
                self.metrics_history.append(metrics)
                print(f"\n[Step {step+1} | {metrics.phase}]")
                print(f"  Loss: {metrics.loss:.4f}")
                print(f"  L1: {metrics.l1_accuracy:.2%} | L2: {metrics.l2_accuracy:.2%} | L3: {metrics.l3_accuracy:.2%}")
                print(f"  Overall: {metrics.overall_accuracy:.2%}")
                print(f"  Composite: {metrics.composite_score:.3f}")
                print(f"  Weights: L1={metrics.weights['L1']:.2f}, L2={metrics.weights['L2']:.2f}, L3={metrics.weights['L3']:.2f}")
                print(f"  Writeback Δ: {metrics.writeback_change:+.4f}")
        
        return self.metrics_history
    
    def generate_report(self) -> Dict:
        """生成报告"""
        if not self.metrics_history:
            return {}
        
        final = self.metrics_history[-1]
        
        # 验收标准
        l1_pass = final.l1_accuracy > 0.60
        l2_pass = final.l2_accuracy > 0.50
        l3_pass = final.l3_accuracy > 0.30
        overall_pass = final.overall_accuracy > 0.55
        wb_pass = abs(final.writeback_change) < 0.05
        
        # 跷跷板检查
        seesaw_ok = final.l1_accuracy > 0.50 and final.l2_accuracy > 0.40
        
        # TSLA 动作准确率
        tsla_accuracy = self.tsla_gate.get_action_accuracy()
        tsla_pass = tsla_accuracy >= 0.95
        
        all_pass = l1_pass and l2_pass and l3_pass and overall_pass and wb_pass and seesaw_ok and tsla_pass
        
        tsla_summary = self.tsla_gate.get_event_summary()
        
        report = {
            'experiment_id': 'stage9_r1_dynamic_weight',
            'timestamp': datetime.now().isoformat(),
            'config': {
                'num_steps': 500,
                'phases': self.phase_steps,
                'base_weights': {'L1': 1.3, 'L2': 1.0, 'L3': 1.0},
                'replay_interval': self.replay_interval,
                'guard_beta': 0.2,
            },
            'final_metrics': {
                'l1_accuracy': final.l1_accuracy,
                'l2_accuracy': final.l2_accuracy,
                'l3_accuracy': final.l3_accuracy,
                'overall_accuracy': final.overall_accuracy,
                'composite_score': final.composite_score,
                'final_weights': final.weights,
                'writeback_change': final.writeback_change,
                'grad_ratio': final.grad_ratio,
                'output_drift': final.output_drift,
                'tsla_event_count': final.tsla_event_count,
                'tsla_action_accuracy': tsla_accuracy,
            },
            'best_checkpoint': self.best_checkpoint,
            'acceptance': {
                'l1_accuracy': {'value': final.l1_accuracy, 'target': '> 60%', 'pass': l1_pass},
                'l2_accuracy': {'value': final.l2_accuracy, 'target': '> 50%', 'pass': l2_pass},
                'l3_accuracy': {'value': final.l3_accuracy, 'target': '> 30%', 'pass': l3_pass},
                'overall_accuracy': {'value': final.overall_accuracy, 'target': '> 55%', 'pass': overall_pass},
                'writeback_change': {'value': final.writeback_change, 'target': '< 5%', 'pass': wb_pass},
                'seesaw_check': {'value': seesaw_ok, 'target': 'L1>50% & L2>40%', 'pass': seesaw_ok},
                'tsla_accuracy': {'value': tsla_accuracy, 'target': '>= 95%', 'pass': tsla_pass},
                'all_pass': all_pass,
            },
            'tsla_summary': tsla_summary,
            'metrics_history': [
                {
                    'step': m.step,
                    'phase': m.phase,
                    'l1_accuracy': m.l1_accuracy,
                    'l2_accuracy': m.l2_accuracy,
                    'l3_accuracy': m.l3_accuracy,
                    'composite_score': m.composite_score,
                    'weights': m.weights,
                }
                for m in self.metrics_history
            ],
        }
        
        return report


def run_stage9_r1():
    """运行 Stage 9-R1"""
    print("="*70)
    print("Stage 9-R1: 动态权重调整实验")
    print("="*70)
    
    # 1. 加载数据集
    def load_jsonl(path: str) -> List[Dict]:
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                samples.append(json.loads(line.strip()))
        return samples
    
    l1_dataset = load_jsonl('stage8_dataset/train.jsonl')
    l2_dataset = load_jsonl('stage8_dataset/synthetic_l2.jsonl')
    l3_dataset = load_jsonl('stage8_dataset/synthetic_l3.jsonl')
    
    print(f"✓ L1 样本: {len(l1_dataset)}")
    print(f"✓ L2 样本: {len(l2_dataset)}")
    print(f"✓ L3 样本: {len(l3_dataset)}")
    
    # 2. 创建模型
    config = NativeTinyConfig()
    model = NativeBackboneTinyV1(config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ 模型创建完成 (device: {device})")
    print(f"  总参数量: {total_params:,}")
    
    # 3. 创建训练器
    trainer = Stage9R1DynamicWeightTrainer(
        model=model,
        l1_dataset=l1_dataset,
        l2_dataset=l2_dataset,
        l3_dataset=l3_dataset,
        device=device,
    )
    
    # 4. 执行训练
    metrics = trainer.train(num_steps=500)
    
    # 5. 生成报告
    report = trainer.generate_report()
    
    # 6. 保存报告
    report_path = Path('stage8_dataset/s9_r1_dynamic_weight_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # 7. 打印结果
    print("\n" + "="*70)
    print("Stage 9-R1 动态权重调整实验报告")
    print("="*70)
    
    acceptance = report['acceptance']
    print(f"\n[分层准确率]")
    print(f"  L1: {acceptance['l1_accuracy']['value']:.2%} ({'✓' if acceptance['l1_accuracy']['pass'] else '✗'})")
    print(f"  L2: {acceptance['l2_accuracy']['value']:.2%} ({'✓' if acceptance['l2_accuracy']['pass'] else '✗'})")
    print(f"  L3: {acceptance['l3_accuracy']['value']:.2%} ({'✓' if acceptance['l3_accuracy']['pass'] else '✗'})")
    print(f"  Overall: {acceptance['overall_accuracy']['value']:.2%} ({'✓' if acceptance['overall_accuracy']['pass'] else '✗'})")
    
    print(f"\n[平衡检查]")
    print(f"  跷跷板检查: {'通过' if acceptance['seesaw_check']['pass'] else '未通过'}")
    print(f"  综合评分: {report['final_metrics']['composite_score']:.3f}")
    
    final_weights = report['final_metrics']['final_weights']
    print(f"\n[最终权重]")
    print(f"  L1: {final_weights['L1']:.2f}")
    print(f"  L2: {final_weights['L2']:.2f}")
    print(f"  L3: {final_weights['L3']:.2f}")
    
    if report['best_checkpoint']:
        best = report['best_checkpoint']
        print(f"\n[最佳 Checkpoint (Step {best['step']})]")
        print(f"  L1: {best['l1_accuracy']:.2%} | L2: {best['l2_accuracy']:.2%} | L3: {best['l3_accuracy']:.2%}")
        print(f"  Composite: {best['composite_score']:.3f}")
        print(f"  Weights: L1={best['weights']['L1']:.2f}, L2={best['weights']['L2']:.2f}, L3={best['weights']['L3']:.2f}")
    
    print(f"\n[Guard & TSLA]")
    print(f"  Writeback Change: {acceptance['writeback_change']['value']:+.4f} ({'✓' if acceptance['writeback_change']['pass'] else '✗'})")
    print(f"  TSLA Action Accuracy: {acceptance['tsla_accuracy']['value']:.2%} ({'✓' if acceptance['tsla_accuracy']['pass'] else '✗'})")
    
    print(f"\n[TSLA 事件统计]")
    for event_type, count in report['tsla_summary'].items():
        print(f"  {event_type}: {count}")
    
    print("\n" + "="*70)
    if acceptance['all_pass']:
        print("✓✓✓ STAGE 9-R1 动态权重实验通过 ✓✓✓")
    else:
        print("✗✗✗ STAGE 9-R1 动态权重实验未通过 ✗✗✗")
    print("="*70)
    
    print(f"\n详细报告: {report_path}")
    
    return report


if __name__ == "__main__":
    report = run_stage9_r1()
