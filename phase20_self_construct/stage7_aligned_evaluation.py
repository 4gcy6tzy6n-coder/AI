"""
Stage 7 对齐评估模块

解决50-step实验和100-step主线验收的评估口径不一致问题

核心原则:
1. 同一个writeback指标定义
2. 同一个评估输入(固定样本集)
3. 同一个基线对照
4. 同一个随机性控制
5. 同一个聚合方式
"""

import torch
import torch.nn.functional as F
from typing import Dict, List
import hashlib


# 固定评估配置
EVAL_CONFIG = {
    'num_samples': 30,
    'seed': 42,
    'input_length': 50,
    'vocab_size': 10000,
}


class AlignedWritebackEvaluator:
    """
    对齐的Writeback评估器
    
    所有评估必须通过这个类进行，确保口径一致
    """
    
    def __init__(self, num_samples: int = None, seed: int = None):
        self.num_samples = num_samples or EVAL_CONFIG['num_samples']
        self.seed = seed or EVAL_CONFIG['seed']
        
        # 生成固定评估样本 (只生成一次，终身复用)
        torch.manual_seed(self.seed)
        self.fixed_samples = [
            torch.randint(0, EVAL_CONFIG['vocab_size'], (1, EVAL_CONFIG['input_length']))
            for _ in range(self.num_samples)
        ]
        
        # 缓存基线输出
        self.baseline_outputs: List[torch.Tensor] = None
        self.baseline_hash: str = None
    
    def set_baseline(self, model):
        """设置基线 (训练前调用一次)"""
        model.eval()
        self.baseline_outputs = []
        
        with torch.no_grad():
            for input_ids in self.fixed_samples:
                outputs = model(input_ids)
                wb_logits = outputs['writeback_logits']
                self.baseline_outputs.append(wb_logits.clone())
        
        # 记录基线模型哈希
        self.baseline_hash = self._compute_model_hash(model)
        
        # 计算基线score
        baseline_score = self._compute_score_from_logits(self.baseline_outputs)
        
        return {
            'hash': self.baseline_hash,
            'score': baseline_score,
            'num_samples': len(self.fixed_samples),
        }
    
    def evaluate(self, model) -> Dict:
        """
        评估writeback能力
        
        Returns:
            包含以下字段的字典:
            - score: 当前writeback score
            - baseline_score: 基线score
            - change: 相对变化 (current - baseline)
            - change_pct: 百分比变化
            - kl: 输出KL散度
            - model_hash: 当前模型哈希
        """
        if self.baseline_outputs is None:
            raise RuntimeError("Must call set_baseline() before evaluate()")
        
        model.eval()
        current_outputs = []
        
        with torch.no_grad():
            for input_ids in self.fixed_samples:
                outputs = model(input_ids)
                wb_logits = outputs['writeback_logits']
                current_outputs.append(wb_logits.clone())
        
        # 计算scores
        current_score = self._compute_score_from_logits(current_outputs)
        baseline_score = self._compute_score_from_logits(self.baseline_outputs)
        
        # 计算变化
        change = current_score - baseline_score
        change_pct = change / baseline_score if baseline_score > 0 else 0
        
        # 计算KL
        kl = self._compute_kl(self.baseline_outputs, current_outputs)
        
        return {
            'score': current_score,
            'baseline_score': baseline_score,
            'change': change,
            'change_pct': change_pct,
            'kl': kl,
            'model_hash': self._compute_model_hash(model),
            'baseline_hash': self.baseline_hash,
        }
    
    def _compute_score_from_logits(self, logits_list: List[torch.Tensor]) -> float:
        """从logits计算writeback score"""
        scores = []
        for logits in logits_list:
            probs = F.softmax(logits, dim=-1)
            # 取类别1的概率作为writeback score
            score = probs[0, 1].item() if probs.shape[1] > 1 else probs[0, 0].item()
            scores.append(score)
        return sum(scores) / len(scores)
    
    def _compute_kl(self, baseline_logits: List[torch.Tensor], current_logits: List[torch.Tensor]) -> float:
        """计算KL散度"""
        total_kl = 0
        for base_logits, curr_logits in zip(baseline_logits, current_logits):
            kl = F.kl_div(
                F.log_softmax(curr_logits, dim=-1),
                F.softmax(base_logits, dim=-1),
                reduction='batchmean'
            ).item()
            total_kl += kl
        return total_kl / len(baseline_logits)
    
    def _compute_model_hash(self, model) -> str:
        """计算模型哈希"""
        params = []
        for param in model.parameters():
            params.append(param.data.cpu().numpy().tobytes())
        combined = b''.join(params)
        return hashlib.md5(combined).hexdigest()[:16]


class StepByStepMonitor:
    """
    逐步监控器
    
    在20/40/60/80/100步记录两套指标:
    1. 固定样本writeback score
    2. 主线协议writeback score
    """
    
    def __init__(self, aligned_evaluator: AlignedWritebackEvaluator, protocol_evaluator=None):
        self.aligned_evaluator = aligned_evaluator
        self.protocol_evaluator = protocol_evaluator
        self.history = []
    
    def record(self, step: int, model, protocol=None) -> Dict:
        """记录当前step的指标"""
        # 固定样本评估
        aligned_result = self.aligned_evaluator.evaluate(model)
        
        # 协议评估 (如果提供)
        protocol_result = None
        if protocol is not None:
            eval_result = protocol.evaluate_with_protocol(f'step_{step}', num_samples=30)
            baseline = protocol.baseline.scores
            protocol_result = {
                'writeback_score': eval_result.scores.writeback_score,
                'baseline_writeback': baseline.writeback_score,
                'change': eval_result.scores.writeback_score - baseline.writeback_score,
            }
        
        record = {
            'step': step,
            'aligned': aligned_result,
            'protocol': protocol_result,
        }
        
        self.history.append(record)
        return record
    
    def print_comparison(self):
        """打印对比表"""
        print("\n" + "="*80)
        print("Step-by-Step 监控对比")
        print("="*80)
        print(f"{'Step':<6} {'Aligned Score':<15} {'Aligned Δ':<12} {'Protocol Score':<15} {'Protocol Δ':<12} {'Diff':<10}")
        print("-"*80)
        
        for record in self.history:
            step = record['step']
            aligned = record['aligned']
            protocol = record['protocol']
            
            aligned_str = f"{aligned['score']:.4f}"
            aligned_change_str = f"{aligned['change']:+.4f}"
            
            if protocol:
                protocol_str = f"{protocol['writeback_score']:.4f}"
                protocol_change_str = f"{protocol['change']:+.4f}"
                diff = aligned['change'] - protocol['change']
                diff_str = f"{diff:+.4f}"
            else:
                protocol_str = "N/A"
                protocol_change_str = "N/A"
                diff_str = "N/A"
            
            print(f"{step:<6} {aligned_str:<15} {aligned_change_str:<12} {protocol_str:<15} {protocol_change_str:<12} {diff_str:<10}")
        
        print("="*80)


# 便捷函数
def create_aligned_evaluator(model) -> AlignedWritebackEvaluator:
    """创建并初始化对齐评估器"""
    evaluator = AlignedWritebackEvaluator()
    baseline_info = evaluator.set_baseline(model)
    print(f"[AlignedEval] 基线已设置: score={baseline_info['score']:.4f}, hash={baseline_info['hash']}")
    return evaluator


if __name__ == "__main__":
    print("="*70)
    print("对齐评估模块测试")
    print("="*70)
    
    # 创建简单模型
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = torch.nn.Embedding(10000, 128)
            self.writeback_head = torch.nn.Linear(128, 2)
        
        def forward(self, input_ids):
            x = self.embeddings(input_ids).mean(dim=1)
            writeback_logits = self.writeback_head(x)
            return {'writeback_logits': writeback_logits}
    
    model = DummyModel()
    
    # 创建评估器
    evaluator = create_aligned_evaluator(model)
    
    # 模拟训练后评估
    print("\n模拟训练后评估...")
    
    # 修改模型参数
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param) * 0.01)
    
    result = evaluator.evaluate(model)
    
    print(f"当前score: {result['score']:.4f}")
    print(f"基线score: {result['baseline_score']:.4f}")
    print(f"变化: {result['change']:+.4f} ({result['change_pct']:+.2%})")
    print(f"KL: {result['kl']:.6f}")
    print(f"模型hash: {result['model_hash']}")
    
    print("\n✓ 测试完成")
