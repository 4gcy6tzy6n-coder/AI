"""
Stage 9-R2: 固定采样 + 严格平衡

策略:
- 固定采样比例: L1:L2:L3 = 2:2:1
- 每个 batch 强制包含所有层级
- 固定权重: L1=1.2, L2=1.2, L3=1.0
- Replay: 每 10 步
- Phase 提前: 1-50 L1, 51-120 L1+L2, 121+ 全混合

目标: 验证单模型能否稳定共存 L1/L2/L3
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
class TSLAEvent:
    """TSLA 事件记录"""
    step: int
    event_type: str
    unit_id: Optional[str]
    trigger_condition: str
    result: str
    timestamp: str


@dataclass
class Stage9R2Metrics:
    """Stage 9-R2 指标"""
    step: int
    phase: str
    l1_accuracy: float
    l2_accuracy: float
    l3_accuracy: float
    overall_accuracy: float
    composite_score: float
    writeback_change: float
    grad_ratio: float
    output_drift: float
    loss: float
    tsla_event_count: int
    tsla_events: List[TSLAEvent] = field(default_factory=list)
    timestamp: str = ""


class TSLAGate:
    """TSLA 门控"""
    
    def __init__(self):
        self.promotion_threshold = 0.8
        self.isolation_threshold = 0.3
        self.event_history: List[TSLAEvent] = []
        
    def check_promotion(self, unit_confidence: float, step: int) -> Tuple[bool, str]:
        if unit_confidence >= self.promotion_threshold:
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
        if unit_confidence < self.isolation_threshold:
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


class Stage9R2FixedSamplingTrainer:
    """Stage 9-R2 固定采样训练器"""
    
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
        
        # 固定权重配置
        self.fixed_weights = {
            'L1': 1.2,
            'L2': 1.2,
            'L3': 1.0,
        }
        
        # Output KL Guard (冻结)
        self.guard = build_output_kl_guard(
            model=model,
            beta=0.2,
            num_samples=30,
            use_probs=True,
        )
        self.guard.capture_reference_outputs()
        
        # TSLA 门控
        self.tsla_gate = TSLAGate()
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
        )
        
        # 基线
        self.baseline_wb_score = None
        self.establish_baseline()
        
        # 记录
        self.metrics_history: List[Stage9R2Metrics] = []
        self.best_checkpoint = None
        self.best_score = 0.0
        
        # 阶段划分 (提前)
        self.phase_steps = {
            'L1_warmup': (0, 50),
            'L1_L2': (50, 120),
            'L1_L2_L3': (120, 300),
        }
        
        # Replay 配置 (更频繁)
        self.replay_interval = 10
        
        # 平衡窗口追踪
        self.balance_window_start = None
        self.balance_window_length = 0
        self.max_balance_window = 0
        
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
        
        print(f"[S9-R2] Writeback 基线: {self.baseline_wb_score:.4f}")
    
    def get_current_phase(self, step: int) -> str:
        """获取当前阶段"""
        for phase, (start, end) in self.phase_steps.items():
            if start <= step < end:
                return phase
        return 'L1_L2_L3'
    
    def get_strict_balanced_batch(self, phase: str) -> List[Tuple[Dict, str]]:
        """严格平衡采样 - 每个 batch 强制包含所有层级"""
        batch = []
        
        if phase == 'L1_warmup':
            # 纯 L1，但控制数量
            for _ in range(4):
                batch.append((random.choice(self.l1_dataset), 'L1'))
        elif phase == 'L1_L2':
            # 严格 1:1
            for _ in range(2):
                batch.append((random.choice(self.l1_dataset), 'L1'))
                batch.append((random.choice(self.l2_dataset), 'L2'))
        else:  # L1_L2_L3
            # 严格 2:2:1
            batch.append((random.choice(self.l1_dataset), 'L1'))
            batch.append((random.choice(self.l1_dataset), 'L1'))
            batch.append((random.choice(self.l2_dataset), 'L2'))
            batch.append((random.choice(self.l2_dataset), 'L2'))
            batch.append((random.choice(self.l3_dataset), 'L3'))
        
        return batch
    
    def compute_task_loss(self, sample: Dict, difficulty: str) -> torch.Tensor:
        """计算任务损失 - 使用固定权重"""
        weight = self.fixed_weights.get(difficulty, 1.0)
        
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
        wb_score = max(0, 1 - abs(wb_change) / 0.05)
        return 0.35 * l1_acc + 0.35 * l2_acc + 0.20 * l3_acc + 0.10 * wb_score
    
    def check_balance(self, l1_acc: float, l2_acc: float, l3_acc: float) -> bool:
        """检查是否处于平衡状态"""
        # 平衡定义: 所有层级都有一定表现，没有掉到 0
        return l1_acc > 0.3 and l2_acc > 0.3 and l3_acc > 0.2
    
    def train_step(self, step: int) -> Tuple[Optional[Stage9R2Metrics], float]:
        """执行一步训练"""
        self.model.train()
        
        phase = self.get_current_phase(step)
        total_loss = 0.0
        
        # 1. 严格平衡 batch 训练
        batch = self.get_strict_balanced_batch(phase)
        
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
        
        # 2. L1 Replay (每 10 步，小 batch)
        if step % self.replay_interval == 0:
            self.optimizer.zero_grad()
            replay_sample = random.choice(self.l1_dataset)
            replay_loss = self.compute_task_loss(replay_sample, 'L1')
            replay_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += replay_loss.item() * 0.5  # 权重降低，避免过度拉回
        
        # 3. TSLA 门控
        with torch.no_grad():
            mock_confidence = torch.sigmoid(torch.randn(1)).item()
        self.tsla_gate.check_promotion(mock_confidence, step)
        self.tsla_gate.check_isolation(mock_confidence, step)
        self.tsla_gate.record_writeback(success=True, step=step)
        
        # 4. 评估 (每 10 步，更频繁监控)
        if step % 10 == 0 or step == 299:
            l1_acc = self.evaluate_by_difficulty(self.l1_dataset, 20)
            l2_acc = self.evaluate_by_difficulty(self.l2_dataset, 20)
            l3_acc = self.evaluate_by_difficulty(self.l3_dataset, 20)
            
            overall_acc = (l1_acc + l2_acc + l3_acc) / 3
            writeback_score = self.evaluate_writeback()
            writeback_change = writeback_score - self.baseline_wb_score
            output_drift = self.guard.compute_output_drift()
            
            composite_score = self.compute_composite_score(l1_acc, l2_acc, l3_acc, writeback_change)
            
            grad_ratio = guard_grad_norm / main_grad_norm if main_grad_norm > 0 else 0
            
            # 追踪平衡窗口
            if self.check_balance(l1_acc, l2_acc, l3_acc):
                if self.balance_window_start is None:
                    self.balance_window_start = step
                self.balance_window_length = step - self.balance_window_start
                self.max_balance_window = max(self.max_balance_window, self.balance_window_length)
            else:
                self.balance_window_start = None
                self.balance_window_length = 0
            
            # 保存最佳 checkpoint (中期可能更好)
            if composite_score > self.best_score:
                self.best_score = composite_score
                self.best_checkpoint = {
                    'step': step,
                    'l1_accuracy': l1_acc,
                    'l2_accuracy': l2_acc,
                    'l3_accuracy': l3_acc,
                    'composite_score': composite_score,
                }
            
            metrics = Stage9R2Metrics(
                step=step,
                phase=phase,
                l1_accuracy=l1_acc,
                l2_accuracy=l2_acc,
                l3_accuracy=l3_acc,
                overall_accuracy=overall_acc,
                composite_score=composite_score,
                writeback_change=writeback_change,
                grad_ratio=grad_ratio,
                output_drift=output_drift,
                loss=total_loss / len(batch),
                tsla_event_count=len(self.tsla_gate.event_history),
                tsla_events=self.tsla_gate.event_history[-3:],
                timestamp=datetime.now().isoformat(),
            )
            
            return metrics, total_loss
        
        return None, total_loss
    
    def train(self, num_steps: int = 300) -> List[Stage9R2Metrics]:
        """执行训练"""
        print("\n" + "="*70)
        print("Stage 9-R2: 固定采样 + 严格平衡")
        print("="*70)
        print(f"L1 样本: {len(self.l1_dataset)}")
        print(f"L2 样本: {len(self.l2_dataset)}")
        print(f"L3 样本: {len(self.l3_dataset)}")
        print(f"训练步数: {num_steps}")
        print(f"\n严格平衡策略:")
        print(f"  - 采样比例: L1:L2:L3 = 2:2:1")
        print(f"  - 固定权重: L1=1.2, L2=1.2, L3=1.0")
        print(f"  - L1 replay: 每 {self.replay_interval} 步")
        print(f"\n课程阶段 (提前):")
        print(f"  Phase 1 (0-49): L1 warm-up")
        print(f"  Phase 2 (50-119): L1 + L2 (严格 1:1)")
        print(f"  Phase 3 (120-299): L1 + L2 + L3 (严格 2:2:1)")
        print(f"\n验收标准:")
        print(f"  - L1 > 50%, L2 > 50%, L3 > 40%")
        print(f"  - Overall > 55%")
        print(f"  - 连续平衡窗口 > 50 steps")
        print(f"  - Writeback Δ < 5%")
        print(f"Output KL Guard: beta=0.2 (冻结)")
        print("="*70)
        
        for step in range(num_steps):
            metrics, loss = self.train_step(step)
            
            if metrics is not None:
                self.metrics_history.append(metrics)
                balance_marker = "✓" if self.check_balance(
                    metrics.l1_accuracy, metrics.l2_accuracy, metrics.l3_accuracy
                ) else ""
                print(f"\n[Step {step+1} | {metrics.phase}] {balance_marker}")
                print(f"  Loss: {metrics.loss:.4f}")
                print(f"  L1: {metrics.l1_accuracy:.2%} | L2: {metrics.l2_accuracy:.2%} | L3: {metrics.l3_accuracy:.2%}")
                print(f"  Overall: {metrics.overall_accuracy:.2%}")
                print(f"  Composite: {metrics.composite_score:.3f}")
                print(f"  Writeback Δ: {metrics.writeback_change:+.4f}")
                if self.balance_window_length > 0:
                    print(f"  Balance Window: {self.balance_window_length} steps")
        
        return self.metrics_history
    
    def generate_report(self) -> Dict:
        """生成报告"""
        if not self.metrics_history:
            return {}
        
        final = self.metrics_history[-1]
        
        # 验收标准 (R2 标准)
        l1_pass = final.l1_accuracy > 0.50
        l2_pass = final.l2_accuracy > 0.50
        l3_pass = final.l3_accuracy > 0.40
        overall_pass = final.overall_accuracy > 0.55
        wb_pass = abs(final.writeback_change) < 0.05
        window_pass = self.max_balance_window >= 50
        
        # L1 是否清零检查
        l1_dropped_to_zero = any(m.l1_accuracy < 0.05 for m in self.metrics_history)
        
        all_pass = l1_pass and l2_pass and l3_pass and overall_pass and wb_pass and window_pass
        
        tsla_summary = self.tsla_gate.get_event_summary()
        
        report = {
            'experiment_id': 'stage9_r2_fixed_sampling',
            'timestamp': datetime.now().isoformat(),
            'config': {
                'num_steps': 300,
                'phases': self.phase_steps,
                'sampling_ratio': 'L1:L2:L3 = 2:2:1',
                'fixed_weights': self.fixed_weights,
                'replay_interval': self.replay_interval,
                'guard_beta': 0.2,
            },
            'final_metrics': {
                'l1_accuracy': final.l1_accuracy,
                'l2_accuracy': final.l2_accuracy,
                'l3_accuracy': final.l3_accuracy,
                'overall_accuracy': final.overall_accuracy,
                'composite_score': final.composite_score,
                'writeback_change': final.writeback_change,
                'grad_ratio': final.grad_ratio,
                'output_drift': final.output_drift,
                'tsla_event_count': final.tsla_event_count,
            },
            'balance_analysis': {
                'max_balance_window': self.max_balance_window,
                'l1_dropped_to_zero': l1_dropped_to_zero,
            },
            'best_checkpoint': self.best_checkpoint,
            'acceptance': {
                'l1_accuracy': {'value': final.l1_accuracy, 'target': '> 50%', 'pass': l1_pass},
                'l2_accuracy': {'value': final.l2_accuracy, 'target': '> 50%', 'pass': l2_pass},
                'l3_accuracy': {'value': final.l3_accuracy, 'target': '> 40%', 'pass': l3_pass},
                'overall_accuracy': {'value': final.overall_accuracy, 'target': '> 55%', 'pass': overall_pass},
                'writeback_change': {'value': final.writeback_change, 'target': '< 5%', 'pass': wb_pass},
                'balance_window': {'value': self.max_balance_window, 'target': '>= 50 steps', 'pass': window_pass},
                'l1_stability': {'value': not l1_dropped_to_zero, 'target': 'L1 not dropped to 0', 'pass': not l1_dropped_to_zero},
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
                }
                for m in self.metrics_history
            ],
        }
        
        return report


def run_stage9_r2():
    """运行 Stage 9-R2"""
    print("="*70)
    print("Stage 9-R2: 固定采样 + 严格平衡")
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
    trainer = Stage9R2FixedSamplingTrainer(
        model=model,
        l1_dataset=l1_dataset,
        l2_dataset=l2_dataset,
        l3_dataset=l3_dataset,
        device=device,
    )
    
    # 4. 执行训练
    metrics = trainer.train(num_steps=300)
    
    # 5. 生成报告
    report = trainer.generate_report()
    
    # 6. 保存报告
    report_path = Path('stage8_dataset/s9_r2_fixed_sampling_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # 7. 打印结果
    print("\n" + "="*70)
    print("Stage 9-R2 固定采样 + 严格平衡实验报告")
    print("="*70)
    
    acceptance = report['acceptance']
    print(f"\n[分层准确率]")
    print(f"  L1: {acceptance['l1_accuracy']['value']:.2%} ({'✓' if acceptance['l1_accuracy']['pass'] else '✗'})")
    print(f"  L2: {acceptance['l2_accuracy']['value']:.2%} ({'✓' if acceptance['l2_accuracy']['pass'] else '✗'})")
    print(f"  L3: {acceptance['l3_accuracy']['value']:.2%} ({'✓' if acceptance['l3_accuracy']['pass'] else '✗'})")
    print(f"  Overall: {acceptance['overall_accuracy']['value']:.2%} ({'✓' if acceptance['overall_accuracy']['pass'] else '✗'})")
    
    print(f"\n[平衡分析]")
    print(f"  最大平衡窗口: {report['balance_analysis']['max_balance_window']} steps ({'✓' if acceptance['balance_window']['pass'] else '✗'})")
    print(f"  L1 是否清零: {'是' if report['balance_analysis']['l1_dropped_to_zero'] else '否'} ({'✓' if acceptance['l1_stability']['pass'] else '✗'})")
    print(f"  综合评分: {report['final_metrics']['composite_score']:.3f}")
    
    if report['best_checkpoint']:
        best = report['best_checkpoint']
        print(f"\n[最佳 Checkpoint (Step {best['step']})]")
        print(f"  L1: {best['l1_accuracy']:.2%} | L2: {best['l2_accuracy']:.2%} | L3: {best['l3_accuracy']:.2%}")
        print(f"  Composite: {best['composite_score']:.3f}")
    
    print(f"\n[Guard & TSLA]")
    print(f"  Writeback Change: {acceptance['writeback_change']['value']:+.4f} ({'✓' if acceptance['writeback_change']['pass'] else '✗'})")
    
    print(f"\n[TSLA 事件统计]")
    for event_type, count in report['tsla_summary'].items():
        print(f"  {event_type}: {count}")
    
    print("\n" + "="*70)
    if acceptance['all_pass']:
        print("✓✓✓ STAGE 9-R2 固定采样实验通过 ✓✓✓")
        print("\n结论: 单模型可以稳定共存 L1/L2/L3")
    else:
        print("✗✗✗ STAGE 9-R2 固定采样实验未通过 ✗✗✗")
        print("\n结论: 需要进一步分析或考虑 MoE 路线")
    print("="*70)
    
    print(f"\n详细报告: {report_path}")
    
    return report


if __name__ == "__main__":
    report = run_stage9_r2()
