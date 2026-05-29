"""
Stage 11-A 链路闭环验收脚本

验证完整链路:
问题 → 缺口识别 → 检索决策 → 检索整合 → 候选构建 → TSLA动作 → 记忆动作

验收标准:
1. Gap Detection Accuracy: 该识别缺口的正确识别
2. Retrieval Trigger Accuracy: 该检索时检索，不该时不乱检索
3. TSLA Action Accuracy: 治理动作合理性
4. Memory Action Accuracy: 记忆动作合理性
5. 链路一致性: 各模块决策是否连贯
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import json
import random
from typing import Dict, List, Tuple
from datetime import datetime

from stage11a_self_learning_trainer import (
    SelfLearningModel, SelfLearningSample, Stage11ATrainer,
    Strategy, TSLAAction, MemoryAction
)
from phase19_native_backbone.native_backbone_tiny_v1 import NativeBackboneTinyV1, NativeTinyConfig


class Stage11AValidator:
    """Stage 11-A 链路验收器"""
    
    def __init__(self, model: SelfLearningModel, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.eval()
    
    def validate_sample(self, sample: SelfLearningSample) -> Dict:
        """
        验证单个样本的完整链路
        
        返回详细验证结果
        """
        with torch.no_grad():
            # 编码输入
            input_ids = self._encode(sample.query)
            
            # 前向传播
            outputs = self.model(input_ids)
            
            # 1. 缺口识别验证
            gap_pred = outputs['gap_detection_logits'].argmax(dim=-1).item()
            gap_conf = torch.softmax(outputs['gap_detection_logits'], dim=-1).max().item()
            
            gap_target = sample.model_targets.get('gap_detected', 0)
            gap_correct = (gap_pred == gap_target)
            
            # 2. 检索决策验证
            retrieval_pred = outputs['retrieval_decision_logits'].argmax(dim=-1).item()
            retrieval_conf = torch.softmax(outputs['retrieval_decision_logits'], dim=-1).max().item()
            
            retrieval_target = sample.model_targets.get('retrieval_needed', 0)
            retrieval_correct = (retrieval_pred == retrieval_target)
            
            # 3. 策略选择验证
            strategy_pred = outputs['strategy_logits'].argmax(dim=-1).item()
            strategy_map = ['DIRECT', 'RETRIEVAL_FIRST', 'CONSERVATIVE', 'DECLINE', 'REVIEW']
            strategy_name = strategy_map[strategy_pred] if strategy_pred < len(strategy_map) else 'UNKNOWN'
            
            # 4. TSLA动作验证
            tsla_pred = outputs['tsla_action_logits'].argmax(dim=-1).item()
            tsla_map = ['PASS', 'PROMOTE', 'ISOLATE', 'ERROR_ARCHIVE', 'DOWNGRADE', 'REFLOW', 'SPLIT', 'EXCLUDE']
            tsla_name = tsla_map[tsla_pred] if tsla_pred < len(tsla_map) else 'UNKNOWN'
            
            tsla_target = sample.model_targets.get('tsla_action', 'PASS')
            tsla_correct = (tsla_name == tsla_target)
            
            # 5. 记忆动作验证
            memory_pred = outputs['memory_action_logits'].argmax(dim=-1).item()
            memory_map = ['NO_WRITE', 'ENTER_REVIEW', 'ISOLATE', 'ERROR_ZONE', 'PROMOTE_CANDIDATE']
            memory_name = memory_map[memory_pred] if memory_pred < len(memory_map) else 'UNKNOWN'
            
            memory_target = sample.model_targets.get('memory_action', 'NO_WRITE')
            memory_correct = (memory_name == memory_target)
            
            # 6. 链路一致性检查
            chain_consistent = self._check_chain_consistency(
                gap_pred, retrieval_pred, strategy_pred, tsla_pred, memory_pred
            )
            
            return {
                'sample_id': sample.id,
                'sample_type': sample.sample_type,
                'query': sample.query,
                
                # 缺口识别
                'gap': {
                    'predicted': gap_pred,
                    'target': gap_target,
                    'confidence': gap_conf,
                    'correct': gap_correct,
                },
                
                # 检索决策
                'retrieval': {
                    'predicted': retrieval_pred,
                    'target': retrieval_target,
                    'confidence': retrieval_conf,
                    'correct': retrieval_correct,
                },
                
                # 策略
                'strategy': {
                    'predicted': strategy_name,
                    'predicted_id': strategy_pred,
                },
                
                # TSLA
                'tsla': {
                    'predicted': tsla_name,
                    'target': tsla_target,
                    'correct': tsla_correct,
                },
                
                # 记忆
                'memory': {
                    'predicted': memory_name,
                    'target': memory_target,
                    'correct': memory_correct,
                },
                
                # 链路一致性
                'chain_consistent': chain_consistent,
            }
    
    def _encode(self, text: str) -> torch.Tensor:
        tokens = [ord(c) % 10000 for c in text[:50]]
        if len(tokens) < 10:
            tokens.extend([0] * (10 - len(tokens)))
        return torch.tensor([tokens], device=self.device)
    
    def _check_chain_consistency(self, gap, retrieval, strategy, tsla, memory) -> bool:
        """检查链路决策是否一致"""
        consistent = True
        
        # 规则1: 如果有缺口，应该考虑检索
        if gap == 1 and retrieval == 0:
            # 有缺口但不检索，可能是CONSERVATIVE策略
            if strategy != 2:  # 2 = CONSERVATIVE
                consistent = False
        
        # 规则2: 如果检索了，策略应该是RETRIEVAL_FIRST
        if retrieval == 1 and strategy != 1:  # 1 = RETRIEVAL_FIRST
            consistent = False
        
        # 规则3: 如果TSLA是ISOLATE，记忆动作应该是隔离相关
        if tsla == 2 and memory not in [2, 3]:  # ISOLATE or ERROR_ZONE
            consistent = False
        
        return consistent
    
    def validate_batch(self, samples: List[SelfLearningSample]) -> Dict:
        """批量验证并计算指标"""
        results = []
        
        for sample in samples:
            result = self.validate_sample(sample)
            results.append(result)
        
        # 计算各项指标
        gap_correct = sum(1 for r in results if r['gap']['correct'])
        retrieval_correct = sum(1 for r in results if r['retrieval']['correct'])
        tsla_correct = sum(1 for r in results if r['tsla']['correct'])
        memory_correct = sum(1 for r in results if r['memory']['correct'])
        chain_consistent = sum(1 for r in results if r['chain_consistent'])
        
        total = len(results)
        
        # 按类型统计
        type_stats = {}
        for sample_type in ['teacher_guided', 'gap_detection', 'retrieval_integration', 'tsla_action', 'memory_governance']:
            type_results = [r for r in results if r['sample_type'] == sample_type]
            if type_results:
                type_stats[sample_type] = {
                    'count': len(type_results),
                    'gap_acc': sum(1 for r in type_results if r['gap']['correct']) / len(type_results),
                    'retrieval_acc': sum(1 for r in type_results if r['retrieval']['correct']) / len(type_results),
                }
        
        return {
            'total_samples': total,
            'gap_accuracy': gap_correct / total if total > 0 else 0,
            'retrieval_accuracy': retrieval_correct / total if total > 0 else 0,
            'tsla_accuracy': tsla_correct / total if total > 0 else 0,
            'memory_accuracy': memory_correct / total if total > 0 else 0,
            'chain_consistency': chain_consistent / total if total > 0 else 0,
            'type_stats': type_stats,
            'detailed_results': results,
        }


def run_validation():
    """运行完整验收"""
    print("="*70)
    print("Stage 11-A 链路闭环验收")
    print("="*70)
    print("验证目标: 问题→缺口识别→检索决策→TSLA动作→记忆动作")
    print("="*70)
    
    # 1. 加载模型
    print("\n[1/4] 加载模型...")
    config = NativeTinyConfig()
    base_model = NativeBackboneTinyV1(config)
    model = SelfLearningModel(base_model)
    
    # 加载检查点
    checkpoint = torch.load('stage8_dataset/stage11a_checkpoint.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("✓ 模型加载完成")
    
    # 2. 加载数据
    print("\n[2/4] 加载数据集...")
    with open('stage8_dataset/stage11a_dataset.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = []
    for item in data:
        sample = SelfLearningSample(
            id=item['id'],
            query=item['query'],
            context=item.get('context', {}),
            teacher_signals=item.get('teacher_signals', {}),
            model_targets=item.get('model_targets', {}),
            sample_type=item.get('sample_type', 'unknown'),
        )
        samples.append(sample)
    
    print(f"✓ 加载 {len(samples)} 条样本")
    
    # 3. 创建验证器
    print("\n[3/4] 创建验证器...")
    validator = Stage11AValidator(model)
    
    # 4. 执行验证
    print("\n[4/4] 执行链路验证...")
    print("="*70)
    
    # 全量验证
    results = validator.validate_batch(samples)
    
    # 打印总体指标
    print("\n【总体指标】")
    print(f"  验证样本数: {results['total_samples']}")
    print(f"  缺口识别准确率: {results['gap_accuracy']:.2%}")
    print(f"  检索决策准确率: {results['retrieval_accuracy']:.2%}")
    print(f"  TSLA动作准确率: {results['tsla_accuracy']:.2%}")
    print(f"  记忆动作准确率: {results['memory_accuracy']:.2%}")
    print(f"  链路一致性: {results['chain_consistency']:.2%}")
    
    # 按类型统计
    print("\n【按类型统计】")
    for sample_type, stats in results['type_stats'].items():
        print(f"  {sample_type}:")
        print(f"    样本数: {stats['count']}")
        print(f"    缺口识别: {stats['gap_acc']:.2%}")
        print(f"    检索决策: {stats['retrieval_acc']:.2%}")
    
    # 5. 抽样详细分析 (20条)
    print("\n" + "="*70)
    print("【抽样详细分析 - 20条】")
    print("="*70)
    
    sample_results = random.sample(results['detailed_results'], min(20, len(results['detailed_results'])))
    
    for i, result in enumerate(sample_results, 1):
        print(f"\n[{i}] {result['sample_id']} ({result['sample_type']})")
        print(f"  问题: {result['query'][:50]}...")
        print(f"  缺口识别: 预测={result['gap']['predicted']}, 目标={result['gap']['target']}, "
              f"置信度={result['gap']['confidence']:.2f}, {'✓' if result['gap']['correct'] else '✗'}")
        print(f"  检索决策: 预测={result['retrieval']['predicted']}, 目标={result['retrieval']['target']}, "
              f"置信度={result['retrieval']['confidence']:.2f}, {'✓' if result['retrieval']['correct'] else '✗'}")
        print(f"  策略选择: {result['strategy']['predicted']}")
        print(f"  TSLA动作: 预测={result['tsla']['predicted']}, 目标={result['tsla']['target']}, "
              f"{'✓' if result['tsla']['correct'] else '✗'}")
        print(f"  记忆动作: 预测={result['memory']['predicted']}, 目标={result['memory']['target']}, "
              f"{'✓' if result['memory']['correct'] else '✗'}")
        print(f"  链路一致性: {'✓' if result['chain_consistent'] else '✗'}")
    
    # 6. 保存结果
    print("\n" + "="*70)
    print("【保存验收报告】")
    print("="*70)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_samples': results['total_samples'],
            'gap_accuracy': results['gap_accuracy'],
            'retrieval_accuracy': results['retrieval_accuracy'],
            'tsla_accuracy': results['tsla_accuracy'],
            'memory_accuracy': results['memory_accuracy'],
            'chain_consistency': results['chain_consistency'],
        },
        'type_stats': results['type_stats'],
        'sample_analysis': [
            {
                'id': r['sample_id'],
                'type': r['sample_type'],
                'gap_correct': r['gap']['correct'],
                'retrieval_correct': r['retrieval']['correct'],
                'tsla_correct': r['tsla']['correct'],
                'memory_correct': r['memory']['correct'],
                'chain_consistent': r['chain_consistent'],
            }
            for r in sample_results
        ],
    }
    
    report_path = 'stage8_dataset/stage11a_validation_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 验收报告已保存: {report_path}")
    
    # 7. 验收结论
    print("\n" + "="*70)
    print("【验收结论】")
    print("="*70)
    
    # 判断标准
    passed = True
    checks = []
    
    if results['gap_accuracy'] >= 0.80:
        checks.append(f"✓ 缺口识别准确率 {results['gap_accuracy']:.1%} ≥ 80%")
    else:
        checks.append(f"✗ 缺口识别准确率 {results['gap_accuracy']:.1%} < 80%")
        passed = False
    
    if results['retrieval_accuracy'] >= 0.75:
        checks.append(f"✓ 检索决策准确率 {results['retrieval_accuracy']:.1%} ≥ 75%")
    else:
        checks.append(f"✗ 检索决策准确率 {results['retrieval_accuracy']:.1%} < 75%")
        passed = False
    
    if results['chain_consistency'] >= 0.70:
        checks.append(f"✓ 链路一致性 {results['chain_consistency']:.1%} ≥ 70%")
    else:
        checks.append(f"✗ 链路一致性 {results['chain_consistency']:.1%} < 70%")
        passed = False
    
    for check in checks:
        print(f"  {check}")
    
    print("\n" + "="*70)
    if passed:
        print("🎉 Stage 11-A 链路闭环验收通过！")
        print("建议: 可以扩展到300条数据继续训练")
    else:
        print("⚠ Stage 11-A 链路闭环验收未完全通过")
        print("建议: 修复问题后再扩数据")
    print("="*70)
    
    return results


if __name__ == "__main__":
    results = run_validation()
