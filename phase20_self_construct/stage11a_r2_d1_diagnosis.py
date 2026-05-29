"""
Stage 11-A-R2-D1: 分型诊断

核心任务: 分析错例模式，找出"稳定地做错"的根因

诊断维度:
1. 按6类样本分型统计
2. gap错误类型分析 (该有判无 / 该无判有)
3. retrieval错误类型分析 (漏检索 / 过度检索)
4. loss覆盖情况检查
5. 样本-标签对齐检查

6类样本分型:
1. 明确无需检索型 - 常识、简单判断
2. 明确需要检索型 - 缺背景、缺项目事实
3. 边界模糊型 - 表面像常识但依赖额外信息
4. 用户错误前提型 - 输入本身带错
5. 冲突/多解型 - 多个合理方向
6. 项目专有知识型 - 与系统/阶段/术语有关
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
from collections import defaultdict

from stage11a_r2_phase1_trainer import (
    SelfLearningModelR2, SelfLearningSample, 
    DeepSeekTeacherR2, TSLA_ACTION_TO_ID, MEMORY_ACTION_TO_ID
)
from phase19_native_backbone.native_backbone_tiny_v1 import NativeBackboneTinyV1, NativeTinyConfig


class SampleClassifier:
    """样本分类器 - 按6类分型"""
    
    @staticmethod
    def classify_sample(query: str, sample_type: str) -> str:
        """将样本分类到6类之一"""
        query_lower = query.lower()
        
        # 1. 明确无需检索型 - 常识、简单判断
        common_knowledge_keywords = [
            '什么是', '什么是人工智能', '什么是机器学习', '什么是深度学习',
            '等于几', '首都是', '由什么组成', '谁写了', '光速', '光合作用',
            'dna', '重力', '2+2', '法国首都', '水是由'
        ]
        for kw in common_knowledge_keywords:
            if kw in query_lower or kw in query:
                return "明确无需检索型"
        
        # 2. 明确需要检索型 - 缺背景、最新信息
        retrieval_keywords = [
            '最新', '当前', '今天', '明天', '最近', '2024', '股价', '天气',
            '诺贝尔奖', 'gpt版本', '突破'
        ]
        for kw in retrieval_keywords:
            if kw in query_lower or kw in query:
                return "明确需要检索型"
        
        # 3. 项目专有知识型
        project_keywords = [
            'stage', '项目', 'baseline', 'guard', 'tsla', '实验', '训练',
            'phase', 'checkpoint', '配置'
        ]
        for kw in project_keywords:
            if kw in query_lower or kw in query:
                return "项目专有知识型"
        
        # 4. 用户错误前提型 - 敏感、错误请求
        error_premise_keywords = [
            '预测彩票', '私人信息', '入侵', '黑客', '虚假陈述', '医疗建议'
        ]
        for kw in error_premise_keywords:
            if kw in query_lower or kw in query:
                return "用户错误前提型"
        
        # 5. 冲突/多解型 - 需要判断、比较
        conflict_keywords = [
            '区别', '比较', 'vs', '和', '还是', '应该', '如何防止', '怎么'
        ]
        for kw in conflict_keywords:
            if kw in query_lower or kw in query:
                return "冲突/多解型"
        
        # 6. 边界模糊型 - 其他
        return "边界模糊型"


class Stage11AR2D1Diagnosis:
    """Stage 11-A-R2-D1 分型诊断器"""
    
    def __init__(self, model: SelfLearningModelR2, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.eval()
        self.classifier = SampleClassifier()
    
    def diagnose_sample(self, sample: SelfLearningSample) -> Dict:
        """诊断单个样本"""
        with torch.no_grad():
            # 编码输入
            input_ids = self._encode(sample.query)
            
            # 前向传播
            outputs = self.model(input_ids)
            
            # 预测结果
            gap_pred = outputs['gap_detection_logits'].argmax(dim=-1).item()
            gap_conf = torch.softmax(outputs['gap_detection_logits'], dim=-1).max().item()
            
            retrieval_pred = outputs['retrieval_decision_logits'].argmax(dim=-1).item()
            retrieval_conf = torch.softmax(outputs['retrieval_decision_logits'], dim=-1).max().item()
            
            # 目标结果
            gap_target = sample.model_targets.get('gap_detected', 0)
            retrieval_target = sample.model_targets.get('retrieval_needed', 0)
            
            # 分类样本
            sample_family = self.classifier.classify_sample(sample.query, sample.sample_type)
            
            # 错误分析
            gap_error_type = None
            if gap_pred != gap_target:
                if gap_target == 1 and gap_pred == 0:
                    gap_error_type = "该有判无 (漏检)"
                elif gap_target == 0 and gap_pred == 1:
                    gap_error_type = "该无判有 (误检)"
            
            retrieval_error_type = None
            if retrieval_pred != retrieval_target:
                if retrieval_target == 1 and retrieval_pred == 0:
                    retrieval_error_type = "漏检索"
                elif retrieval_target == 0 and retrieval_pred == 1:
                    retrieval_error_type = "过度检索"
            
            # 检查gap-retrieval逻辑一致性
            logic_consistent = True
            if gap_pred == 0 and retrieval_pred == 1:
                # 无缺口却检索，可能是合理的(保守策略)
                pass
            elif gap_pred == 1 and retrieval_pred == 0:
                # 有缺口却不检索，逻辑不一致
                logic_consistent = False
            
            return {
                'sample_id': sample.id,
                'sample_type': sample.sample_type,
                'sample_family': sample_family,
                'query': sample.query,
                
                'gap': {
                    'predicted': gap_pred,
                    'target': gap_target,
                    'confidence': gap_conf,
                    'correct': gap_pred == gap_target,
                    'error_type': gap_error_type,
                },
                
                'retrieval': {
                    'predicted': retrieval_pred,
                    'target': retrieval_target,
                    'confidence': retrieval_conf,
                    'correct': retrieval_pred == retrieval_target,
                    'error_type': retrieval_error_type,
                },
                
                'logic_consistent': logic_consistent,
            }
    
    def _encode(self, text: str) -> torch.Tensor:
        tokens = [ord(c) % 10000 for c in text[:50]]
        if len(tokens) < 10:
            tokens.extend([0] * (10 - len(tokens)))
        return torch.tensor([tokens], device=self.device)
    
    def run_full_diagnosis(self, samples: List[SelfLearningSample]) -> Dict:
        """运行完整分型诊断"""
        print("="*70)
        print("Stage 11-A-R2-D1: 分型诊断")
        print("="*70)
        print("诊断目标: 找出'稳定地做错'的根因")
        print("="*70)
        
        # 1. 逐个诊断样本
        print("\n[1/6] 诊断所有样本...")
        diagnoses = []
        for sample in samples:
            diag = self.diagnose_sample(sample)
            diagnoses.append(diag)
        
        print(f"  ✓ 完成 {len(diagnoses)} 条样本诊断")
        
        # 2. 按6类分型统计
        print("\n[2/6] 按6类样本分型统计...")
        family_stats = self._analyze_by_family(diagnoses)
        
        # 3. gap错误类型分析
        print("\n[3/6] Gap错误类型分析...")
        gap_error_analysis = self._analyze_gap_errors(diagnoses)
        
        # 4. retrieval错误类型分析
        print("\n[4/6] Retrieval错误类型分析...")
        retrieval_error_analysis = self._analyze_retrieval_errors(diagnoses)
        
        # 5. 样本-标签对齐检查
        print("\n[5/6] 样本-标签对齐检查...")
        alignment_check = self._check_alignment(diagnoses)
        
        # 6. 错例抽样分析
        print("\n[6/6] 错例抽样详细分析...")
        error_samples = self._analyze_error_samples(diagnoses)
        
        return {
            'total_samples': len(diagnoses),
            'family_stats': family_stats,
            'gap_error_analysis': gap_error_analysis,
            'retrieval_error_analysis': retrieval_error_analysis,
            'alignment_check': alignment_check,
            'error_samples': error_samples,
            'all_diagnoses': diagnoses,
        }
    
    def _analyze_by_family(self, diagnoses: List[Dict]) -> Dict:
        """按6类分型统计"""
        families = defaultdict(lambda: {'total': 0, 'gap_correct': 0, 'retrieval_correct': 0})
        
        for diag in diagnoses:
            family = diag['sample_family']
            families[family]['total'] += 1
            if diag['gap']['correct']:
                families[family]['gap_correct'] += 1
            if diag['retrieval']['correct']:
                families[family]['retrieval_correct'] += 1
        
        print("\n  【6类样本分型统计】")
        for family, stats in families.items():
            gap_acc = stats['gap_correct'] / stats['total'] if stats['total'] > 0 else 0
            retrieval_acc = stats['retrieval_correct'] / stats['total'] if stats['total'] > 0 else 0
            print(f"    {family}:")
            print(f"      样本数: {stats['total']}")
            print(f"      Gap准确率: {gap_acc:.1%}")
            print(f"      Retrieval准确率: {retrieval_acc:.1%}")
        
        return dict(families)
    
    def _analyze_gap_errors(self, diagnoses: List[Dict]) -> Dict:
        """Gap错误类型分析"""
        error_types = defaultdict(int)
        total_errors = 0
        
        for diag in diagnoses:
            if diag['gap']['error_type']:
                error_types[diag['gap']['error_type']] += 1
                total_errors += 1
        
        print("\n  【Gap错误类型分布】")
        print(f"    总错误数: {total_errors}")
        for error_type, count in error_types.items():
            pct = count / total_errors if total_errors > 0 else 0
            print(f"    {error_type}: {count} ({pct:.1%})")
        
        return {
            'total_errors': total_errors,
            'error_distribution': dict(error_types),
        }
    
    def _analyze_retrieval_errors(self, diagnoses: List[Dict]) -> Dict:
        """Retrieval错误类型分析"""
        error_types = defaultdict(int)
        total_errors = 0
        
        for diag in diagnoses:
            if diag['retrieval']['error_type']:
                error_types[diag['retrieval']['error_type']] += 1
                total_errors += 1
        
        print("\n  【Retrieval错误类型分布】")
        print(f"    总错误数: {total_errors}")
        for error_type, count in error_types.items():
            pct = count / total_errors if total_errors > 0 else 0
            print(f"    {error_type}: {count} ({pct:.1%})")
        
        return {
            'total_errors': total_errors,
            'error_distribution': dict(error_types),
        }
    
    def _check_alignment(self, diagnoses: List[Dict]) -> Dict:
        """检查样本-标签对齐"""
        # 检查gap和retrieval的逻辑关系
        consistent_count = 0
        inconsistent_cases = []
        
        for diag in diagnoses:
            gap_target = diag['gap']['target']
            retrieval_target = diag['retrieval']['target']
            
            # 理想逻辑: gap=0时retrieval通常为0，gap=1时retrieval可能为1
            if gap_target == 0 and retrieval_target == 0:
                consistent_count += 1
            elif gap_target == 1 and retrieval_target == 1:
                consistent_count += 1
            elif gap_target == 1 and retrieval_target == 0:
                # 有缺口但不检索，可能是保守策略
                consistent_count += 1
            else:
                # gap=0但retrieval=1，逻辑不一致
                inconsistent_cases.append({
                    'id': diag['sample_id'],
                    'query': diag['query'][:50],
                    'gap_target': gap_target,
                    'retrieval_target': retrieval_target,
                })
        
        alignment_rate = consistent_count / len(diagnoses) if diagnoses else 0
        
        print("\n  【样本-标签对齐检查】")
        print(f"    Gap-Retrieval逻辑一致率: {alignment_rate:.1%}")
        print(f"    不一致样本数: {len(inconsistent_cases)}")
        
        if inconsistent_cases[:5]:
            print("    不一致样例:")
            for case in inconsistent_cases[:5]:
                print(f"      {case['id']}: gap={case['gap_target']}, retrieval={case['retrieval_target']}")
        
        return {
            'alignment_rate': alignment_rate,
            'inconsistent_count': len(inconsistent_cases),
            'inconsistent_cases': inconsistent_cases[:10],
        }
    
    def _analyze_error_samples(self, diagnoses: List[Dict]) -> List[Dict]:
        """错例抽样详细分析"""
        # 找出gap和retrieval都错的样本
        both_wrong = [d for d in diagnoses if not d['gap']['correct'] and not d['retrieval']['correct']]
        
        # 抽样10条
        sample_size = min(10, len(both_wrong))
        sampled = random.sample(both_wrong, sample_size) if both_wrong else []
        
        print("\n  【错例抽样分析 (Gap和Retrieval都错)】")
        print(f"    总错例数: {len(both_wrong)}")
        print(f"    抽样数: {len(sampled)}")
        
        for i, diag in enumerate(sampled, 1):
            print(f"\n    [{i}] {diag['sample_id']} ({diag['sample_family']})")
            print(f"      问题: {diag['query'][:60]}...")
            print(f"      Gap: 预测={diag['gap']['predicted']}, 目标={diag['gap']['target']}, "
                  f"错误={diag['gap']['error_type']}")
            print(f"      Retrieval: 预测={diag['retrieval']['predicted']}, 目标={diag['retrieval']['target']}, "
                  f"错误={diag['retrieval']['error_type']}")
        
        return sampled


def generate_diagnosis_report(diagnosis_result: Dict) -> str:
    """生成诊断报告"""
    report = []
    report.append("="*70)
    report.append("Stage 11-A-R2-D1 分型诊断报告")
    report.append("="*70)
    report.append("")
    
    # 总体情况
    report.append("【一、总体情况】")
    report.append(f"  诊断样本数: {diagnosis_result['total_samples']}")
    report.append("")
    
    # 6类分型统计
    report.append("【二、6类样本分型表现】")
    for family, stats in diagnosis_result['family_stats'].items():
        gap_acc = stats['gap_correct'] / stats['total'] if stats['total'] > 0 else 0
        retrieval_acc = stats['retrieval_correct'] / stats['total'] if stats['total'] > 0 else 0
        report.append(f"  {family}:")
        report.append(f"    样本数: {stats['total']}")
        report.append(f"    Gap准确率: {gap_acc:.1%}")
        report.append(f"    Retrieval准确率: {retrieval_acc:.1%}")
    report.append("")
    
    # Gap错误分析
    report.append("【三、Gap错误分析】")
    gap_analysis = diagnosis_result['gap_error_analysis']
    report.append(f"  总错误数: {gap_analysis['total_errors']}")
    for error_type, count in gap_analysis['error_distribution'].items():
        report.append(f"    {error_type}: {count}")
    report.append("")
    
    # Retrieval错误分析
    report.append("【四、Retrieval错误分析】")
    retrieval_analysis = diagnosis_result['retrieval_error_analysis']
    report.append(f"  总错误数: {retrieval_analysis['total_errors']}")
    for error_type, count in retrieval_analysis['error_distribution'].items():
        report.append(f"    {error_type}: {count}")
    report.append("")
    
    # 对齐检查
    report.append("【五、样本-标签对齐检查】")
    alignment = diagnosis_result['alignment_check']
    report.append(f"  Gap-Retrieval逻辑一致率: {alignment['alignment_rate']:.1%}")
    report.append(f"  不一致样本数: {alignment['inconsistent_count']}")
    report.append("")
    
    # 根因分析
    report.append("【六、根因分析】")
    
    # 分析哪种错误最多
    gap_errors = diagnosis_result['gap_error_analysis']['error_distribution']
    retrieval_errors = diagnosis_result['retrieval_error_analysis']['error_distribution']
    
    if gap_errors.get('该无判有 (误检)', 0) > gap_errors.get('该有判无 (漏检)', 0):
        report.append("  ⚠ 主要问题: Gap过度检测 (该无判有)")
        report.append("    → 模型倾向于认为有缺口，可能过于保守")
    else:
        report.append("  ⚠ 主要问题: Gap漏检 (该有判无)")
        report.append("    → 模型倾向于认为无缺口，可能过于自信")
    
    if retrieval_errors.get('过度检索', 0) > retrieval_errors.get('漏检索', 0):
        report.append("  ⚠ 主要问题: 过度检索")
        report.append("    → 模型倾向于频繁触发检索")
    else:
        report.append("  ⚠ 主要问题: 漏检索")
        report.append("    → 模型倾向于不检索，可能错过重要信息")
    
    report.append("")
    
    # 修复建议
    report.append("【七、修复建议】")
    report.append("  1. 样本设计问题:")
    report.append("     - 缺口识别样本缺少'已知信息状态'上下文")
    report.append("     - 需要让模型基于上下文判断，而不是背题型")
    report.append("")
    report.append("  2. 标签对齐问题:")
    if alignment['alignment_rate'] < 0.9:
        report.append(f"     - Gap-Retrieval逻辑一致率仅{alignment['alignment_rate']:.1%}，需要修正")
    report.append("")
    report.append("  3. 下一步行动:")
    report.append("     - 重写缺口识别样本，增加'known_info'字段")
    report.append("     - 修正gap-retrieval逻辑不一致的样本")
    report.append("     - 重新训练后再次诊断")
    report.append("")
    
    return "\n".join(report)


def run_diagnosis():
    """运行完整诊断流程"""
    print("="*70)
    print("Stage 11-A-R2-D1: 分型诊断启动")
    print("="*70)
    
    # 1. 加载模型
    print("\n[准备] 加载训练好的模型...")
    config = NativeTinyConfig()
    base_model = NativeBackboneTinyV1(config)
    model = SelfLearningModelR2(base_model)
    
    checkpoint = torch.load('stage8_dataset/stage11a_r2_phase1_checkpoint.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("  ✓ 模型加载完成")
    
    # 2. 加载数据
    print("\n[准备] 加载数据集...")
    with open('stage8_dataset/stage11a_r2_phase1_dataset.json', 'r', encoding='utf-8') as f:
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
    
    print(f"  ✓ 加载 {len(samples)} 条样本")
    
    # 3. 创建诊断器并运行
    print("\n[执行] 运行分型诊断...")
    diagnosis = Stage11AR2D1Diagnosis(model)
    result = diagnosis.run_full_diagnosis(samples)
    
    # 4. 生成报告
    print("\n" + "="*70)
    print("生成诊断报告...")
    print("="*70)
    
    report = generate_diagnosis_report(result)
    print(report)
    
    # 5. 保存报告
    report_path = 'stage8_dataset/stage11a_r2_d1_diagnosis_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # 6. 保存详细诊断结果
    result_path = 'stage8_dataset/stage11a_r2_d1_diagnosis_result.json'
    with open(result_path, 'w', encoding='utf-8') as f:
        # 简化保存，避免循环引用
        simplified_result = {
            'total_samples': result['total_samples'],
            'family_stats': result['family_stats'],
            'gap_error_analysis': result['gap_error_analysis'],
            'retrieval_error_analysis': result['retrieval_error_analysis'],
            'alignment_check': {
                'alignment_rate': result['alignment_check']['alignment_rate'],
                'inconsistent_count': result['alignment_check']['inconsistent_count'],
            },
        }
        json.dump(simplified_result, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 诊断报告已保存: {report_path}")
    print(f"✓ 诊断结果已保存: {result_path}")
    
    return result


if __name__ == "__main__":
    result = run_diagnosis()
