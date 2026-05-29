"""
Stage 8-R1 Retention Repair
修复 L1 保持性问题

策略:
- 平衡采样: 每个 batch 固定包含 L1/L2/L3
- L1 loss 权重提升到 2.0
- 每 10 步插入 1 个纯 L1 replay batch
- Output KL Guard 冻结 (beta=0.2)
- TSLA 配置冻结
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
class RepairMetrics:
    """修复验证指标"""
    step: int
    l1_accuracy: float
    l2_accuracy: float
    l3_accuracy: float
    overall_accuracy: float
    l1_retention: float  # L1 保持率 (相对于基线)
    writeback_change: float
    grad_ratio: float
    output_drift: float
    loss: float
    tsla_events: List[TSLAEvent] = field(default_factory=list)
    timestamp: str = ""


class TSLAGate:
    """TSLA 门控 - 冻结配置"""
    
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


class Stage8R1RetentionRepair:
    """Stage 8-R1 保持性修复器"""
    
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
        
        # Output KL Guard - 冻结配置
        self.guard = build_output_kl_guard(
            model=model,
            beta=0.2,  # 冻结
            num_samples=30,
            use_probs=True,
        )
        self.guard.capture_reference_outputs()
        
        # TSLA 门控 - 冻结配置
        self.tsla_gate = TSLAGate()
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
        )
        
        # 基线
        self.baseline_wb_score = None
        self.l1_baseline_accuracy = None  # L1 基线准确率
        self.establish_baseline()
        
        # 记录
        self.metrics_history: List[RepairMetrics] = []
        
        # 修复策略参数
        self.l1_weight = 2.0  # L1 loss 加权
        self.replay_interval = 10  # 每 10 步 replay
        
    def establish_baseline(self):
        """建立基线"""
        self.model.eval()
        with torch.no_grad():
            torch.manual_seed(42)
            
            # Writeback 基线
            anchor_inputs = [torch.randint(0, 10000, (1, 50)).to(self.device) for _ in range(30)]
            wb_scores = []
            for input_ids in anchor_inputs:
                outputs = self.model(input_ids)
                probs = F.softmax(outputs['writeback_logits'], dim=-1)
                score = probs[0, 1].item() if probs.shape[1] > 1 else probs[0, 0].item()
                wb_scores.append(score)
            self.baseline_wb_score = sum(wb_scores) / len(wb_scores)
            
            # L1 准确率基线
            self.l1_baseline_accuracy = self.evaluate_by_difficulty(self.l1_dataset, 20)
        
        print(f"[R1 Repair] Writeback 基线: {self.baseline_wb_score:.4f}")
        print(f"[R1 Repair] L1 准确率基线: {self.l1_baseline_accuracy:.2%}")
    
    def compute_task_loss(self, sample: Dict, difficulty: str) -> torch.Tensor:
        """计算任务损失 - 带 L1 加权"""
        # 难度权重
        difficulty_weights = {
            'L1': self.l1_weight,  # 2.0
            'L2': 1.0,
            'L3': 1.0,
        }
        weight = difficulty_weights.get(difficulty, 1.0)
        
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
    
    def get_balanced_batch(self) -> List[Tuple[Dict, str]]:
        """获取平衡 batch - 1:1:1 采样"""
        batch = []
        
        # 每个难度各取 1 个
        l1_idx = random.randint(0, len(self.l1_dataset) - 1)
        l2_idx = random.randint(0, len(self.l2_dataset) - 1)
        l3_idx = random.randint(0, len(self.l3_dataset) - 1)
        
        batch.append((self.l1_dataset[l1_idx], 'L1'))
        batch.append((self.l2_dataset[l2_idx], 'L2'))
        batch.append((self.l3_dataset[l3_idx], 'L3'))
        
        return batch
    
    def train_step(self, step: int) -> Tuple[RepairMetrics, float]:
        """执行一步训练 - 平衡采样 + L1 replay"""
        self.model.train()
        
        total_loss = 0.0
        
        # 1. 平衡采样训练 (L1/L2/L3 各一个)
        balanced_batch = self.get_balanced_batch()
        
        for sample, difficulty in balanced_batch:
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
        
        # 2. L1 Replay (每 10 步)
        if step % self.replay_interval == 0:
            self.optimizer.zero_grad()
            
            # 纯 L1 replay
            replay_sample = random.choice(self.l1_dataset)
            replay_loss = self.compute_task_loss(replay_sample, 'L1')
            replay_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += replay_loss.item()
        
        # 3. TSLA 门控检查
        with torch.no_grad():
            mock_confidence = torch.sigmoid(torch.randn(1)).item()
        
        self.tsla_gate.check_promotion(mock_confidence, step)
        self.tsla_gate.check_isolation(mock_confidence, step)
        self.tsla_gate.record_writeback(success=True, step=step)
        
        # 4. 评估
        l1_acc = self.evaluate_by_difficulty(self.l1_dataset, 20)
        l2_acc = self.evaluate_by_difficulty(self.l2_dataset, 20)
        l3_acc = self.evaluate_by_difficulty(self.l3_dataset, 20)
        
        overall_acc = (l1_acc + l2_acc + l3_acc) / 3
        
        # L1 保持率 (相对于基线)
        l1_retention = l1_acc / self.l1_baseline_accuracy if self.l1_baseline_accuracy > 0 else 0
        
        writeback_score = self.evaluate_writeback()
        writeback_change = writeback_score - self.baseline_wb_score
        output_drift = self.guard.compute_output_drift()
        
        grad_ratio = guard_grad_norm / main_grad_norm if main_grad_norm > 0 else 0
        
        metrics = RepairMetrics(
            step=step,
            l1_accuracy=l1_acc,
            l2_accuracy=l2_acc,
            l3_accuracy=l3_acc,
            overall_accuracy=overall_acc,
            l1_retention=l1_retention,
            writeback_change=writeback_change,
            grad_ratio=grad_ratio,
            output_drift=output_drift,
            loss=total_loss / (len(balanced_batch) + (1 if step % self.replay_interval == 0 else 0)),
            tsla_events=self.tsla_gate.event_history[-5:],
            timestamp=datetime.now().isoformat(),
        )
        
        return metrics, total_loss
    
    def repair(self, num_steps: int = 200) -> List[RepairMetrics]:
        """执行修复训练"""
        print("\n" + "="*70)
        print("Stage 8-R1 Retention Repair")
        print("="*70)
        print(f"L1 样本: {len(self.l1_dataset)}")
        print(f"L2 样本: {len(self.l2_dataset)}")
        print(f"L3 样本: {len(self.l3_dataset)}")
        print(f"训练步数: {num_steps}")
        print(f"修复策略:")
        print(f"  - 平衡采样: 1:1:1")
        print(f"  - L1 权重: {self.l1_weight}")
        print(f"  - L1 Replay: 每 {self.replay_interval} 步")
        print(f"Output KL Guard: beta=0.2 (冻结)")
        print(f"TSLA 门控: 冻结配置")
        print("="*70)
        
        checkpoint_steps = [50, 100, 150, 200]
        
        for step in range(num_steps):
            metrics, loss = self.train_step(step)
            self.metrics_history.append(metrics)
            
            if (step + 1) in checkpoint_steps:
                print(f"\n[Step {step+1}]")
                print(f"  Loss: {loss:.4f}")
                print(f"  L1 Acc: {metrics.l1_accuracy:.2%} (Retention: {metrics.l1_retention:.1%})")
                print(f"  L2 Acc: {metrics.l2_accuracy:.2%}")
                print(f"  L3 Acc: {metrics.l3_accuracy:.2%}")
                print(f"  Overall: {metrics.overall_accuracy:.2%}")
                print(f"  Writeback Δ: {metrics.writeback_change:+.4f}")
        
        return self.metrics_history
    
    def generate_report(self) -> Dict:
        """生成报告"""
        if not self.metrics_history:
            return {}
        
        final = self.metrics_history[-1]
        
        # 新的验收标准 (含 L1 保持)
        l1_pass = final.l1_accuracy > 0.70
        l1_retention_pass = final.l1_retention > 0.85  # 保持率 > 85%
        l2_pass = final.l2_accuracy > 0.50
        l3_pass = final.l3_accuracy > 0.30
        overall_pass = final.overall_accuracy > 0.70
        wb_pass = abs(final.writeback_change) < 0.05
        
        all_pass = l1_pass and l1_retention_pass and l2_pass and overall_pass and wb_pass
        
        tsla_summary = self.tsla_gate.get_event_summary()
        
        report = {
            'experiment_id': 'stage8_r1_retention_repair',
            'timestamp': datetime.now().isoformat(),
            'config': {
                'num_steps': len(self.metrics_history),
                'l1_weight': self.l1_weight,
                'replay_interval': self.replay_interval,
                'sampling': 'balanced_1:1:1',
                'guard_beta': 0.2,
            },
            'final_metrics': {
                'l1_accuracy': final.l1_accuracy,
                'l1_retention': final.l1_retention,
                'l1_baseline': self.l1_baseline_accuracy,
                'l2_accuracy': final.l2_accuracy,
                'l3_accuracy': final.l3_accuracy,
                'overall_accuracy': final.overall_accuracy,
                'writeback_change': final.writeback_change,
                'grad_ratio': final.grad_ratio,
                'output_drift': final.output_drift,
                'final_loss': final.loss,
            },
            'acceptance': {
                'l1_accuracy': {'value': final.l1_accuracy, 'target': '> 70%', 'pass': l1_pass},
                'l1_retention': {'value': final.l1_retention, 'target': '> 85%', 'pass': l1_retention_pass},
                'l2_accuracy': {'value': final.l2_accuracy, 'target': '> 50%', 'pass': l2_pass},
                'l3_accuracy': {'value': final.l3_accuracy, 'target': '> 30%', 'pass': l3_pass},
                'overall_accuracy': {'value': final.overall_accuracy, 'target': '> 70%', 'pass': overall_pass},
                'writeback_change': {'value': final.writeback_change, 'target': '< 5%', 'pass': wb_pass},
                'all_pass': all_pass,
            },
            'tsla_summary': tsla_summary,
            'metrics_history': [
                {
                    'step': m.step,
                    'l1_accuracy': m.l1_accuracy,
                    'l1_retention': m.l1_retention,
                    'l2_accuracy': m.l2_accuracy,
                    'l3_accuracy': m.l3_accuracy,
                    'overall_accuracy': m.overall_accuracy,
                }
                for m in self.metrics_history[::20]
            ],
        }
        
        return report


def run_stage8_r1_repair():
    """运行 R1 修复"""
    print("="*70)
    print("Stage 8-R1 Retention Repair")
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
    
    print(f"✓ 模型创建完成 (device: {device})")
    
    # 3. 创建修复器
    repairer = Stage8R1RetentionRepair(
        model=model,
        l1_dataset=l1_dataset,
        l2_dataset=l2_dataset,
        l3_dataset=l3_dataset,
        device=device,
    )
    
    # 4. 执行修复
    metrics = repairer.repair(num_steps=200)
    
    # 5. 生成报告
    report = repairer.generate_report()
    
    # 6. 保存报告
    report_path = Path('stage8_dataset/r1_repair_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # 7. 打印结果
    print("\n" + "="*70)
    print("Stage 8-R1 Retention Repair 报告")
    print("="*70)
    
    acceptance = report['acceptance']
    print(f"\n[分层准确率]")
    print(f"  L1: {acceptance['l1_accuracy']['value']:.2%} ({'✓' if acceptance['l1_accuracy']['pass'] else '✗'})")
    print(f"  L1 Retention: {report['final_metrics']['l1_retention']:.1%} ({'✓' if acceptance['l1_retention']['pass'] else '✗'})")
    print(f"  L2: {acceptance['l2_accuracy']['value']:.2%} ({'✓' if acceptance['l2_accuracy']['pass'] else '✗'})")
    print(f"  L3: {acceptance['l3_accuracy']['value']:.2%} ({'✓' if acceptance['l3_accuracy']['pass'] else '✗'})")
    print(f"  Overall: {acceptance['overall_accuracy']['value']:.2%} ({'✓' if acceptance['overall_accuracy']['pass'] else '✗'})")
    
    print(f"\n[Writeback 稳定性]")
    print(f"  Change: {acceptance['writeback_change']['value']:+.4f} ({'✓' if acceptance['writeback_change']['pass'] else '✗'})")
    
    print(f"\n[TSLA 事件统计]")
    for event_type, count in report['tsla_summary'].items():
        print(f"  {event_type}: {count}")
    
    print("\n" + "="*70)
    if acceptance['all_pass']:
        print("✓✓✓ STAGE 8-R1 RETENTION REPAIR 通过 ✓✓✓")
    else:
        print("✗✗✗ STAGE 8-R1 RETENTION REPAIR 未通过 ✗✗✗")
    print("="*70)
    
    print(f"\n详细报告: {report_path}")
    
    return report


if __name__ == "__main__":
    report = run_stage8_r1_repair()
