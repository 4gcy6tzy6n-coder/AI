"""
Stage 11-A-R2-Phase2.5: 破满分验证

核心目标: 证明100%不是假满分，排查捷径学习

验证维度:
1. 模板隔离切分 - 按模板/场景/难度切分，非随机
2. 去捷径字段消融 - 去掉known_info/scenario_type等字段
3. 对抗样本测试 - 加入"脏"样本
4. 标签独立性检查 - 验证TSLA/Memory不是由上游规则派生
5. 人工盲审50条 - 人工判断vs模型预测

判断标准:
- 如果切分后准确率明显掉下去 → 模板记忆
- 如果去掉某字段后准确率暴跌 → 捷径学习
- 如果对抗样本表现差 → 泛化能力不足
- 如果TSLA/Memory和上游高度耦合 → 不是独立能力
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import json
import random
from typing import Dict, List, Tuple
from collections import defaultdict
from dataclasses import dataclass

from phase19_native_backbone.native_backbone_tiny_v1 import NativeBackboneTinyV1, NativeTinyConfig


# ========== 标签映射 ==========
TSLA_ACTION_TO_ID = {
    "保留": 0, "晋升": 1, "隔离": 2, "错误归档": 3,
    "降级": 4, "回流重审": 5, "拆分": 6, "排除": 7,
}

MEMORY_ACTION_TO_ID = {
    "不写入": 0, "进入受审区": 1, "隔离观察": 2,
    "进入错误区": 3, "晋升候选": 4,
}


@dataclass
class ValidationSample:
    """验证样本"""
    id: str
    query: str
    known_info: str
    scenario_type: str
    sample_type: str
    sample_family: str
    difficulty: int
    model_targets: Dict


class Phase25Validator:
    """Phase2.5验证器"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.eval()
    
    def _encode(self, text: str) -> torch.Tensor:
        tokens = [ord(c) % 10000 for c in text[:100]]
        if len(tokens) < 10:
            tokens.extend([0] * (10 - len(tokens)))
        return torch.tensor([tokens], device=self.device)
    
    def evaluate_sample(self, sample: ValidationSample, 
                        use_known_info: bool = True,
                        use_scenario: bool = True) -> Dict:
        """评估单个样本"""
        with torch.no_grad():
            # 构造输入
            input_text = sample.query
            if use_known_info:
                input_text += " | " + sample.known_info
            if use_scenario:
                input_text += f" | 场景:{sample.scenario_type}"
            
            input_ids = self._encode(input_text)
            outputs = self.model(input_ids)
            
            results = {}
            
            # Gap
            gap_pred = outputs['gap_detection_logits'].argmax(dim=-1).item()
            gap_target = sample.model_targets.get('gap_detected', 0)
            results['gap_correct'] = (gap_pred == gap_target)
            
            # Retrieval
            retrieval_pred = outputs['retrieval_decision_logits'].argmax(dim=-1).item()
            retrieval_target = sample.model_targets.get('retrieval_needed', 0)
            results['retrieval_correct'] = (retrieval_pred == retrieval_target)
            
            # TSLA
            if 'tsla_action' in sample.model_targets:
                tsla_pred = outputs['tsla_action_logits'].argmax(dim=-1).item()
                action = sample.model_targets['tsla_action']
                tsla_target = TSLA_ACTION_TO_ID.get(action, 0) if isinstance(action, str) else action
                results['tsla_correct'] = (tsla_pred == tsla_target)
            
            # Memory
            if 'memory_action' in sample.model_targets:
                memory_pred = outputs['memory_action_logits'].argmax(dim=-1).item()
                action = sample.model_targets['memory_action']
                memory_target = MEMORY_ACTION_TO_ID.get(action, 0) if isinstance(action, str) else action
                results['memory_correct'] = (memory_pred == memory_target)
            
            return results


class Phase25ValidationSuite:
    """Phase2.5完整验证套件"""
    
    def __init__(self, model, device='cpu'):
        self.validator = Phase25Validator(model, device)
        self.results = {}
    
    def load_samples(self, dataset_path: str) -> List[ValidationSample]:
        """加载样本"""
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = []
        for item in data:
            sample = ValidationSample(
                id=item['id'],
                query=item['query'],
                known_info=item.get('known_info', ''),
                scenario_type=item.get('scenario_type', 'standard'),
                sample_type=item.get('sample_type', 'unknown'),
                sample_family=item.get('sample_family', 'unknown'),
                difficulty=item.get('difficulty', 1),
                model_targets=item.get('model_targets', {}),
            )
            samples.append(sample)
        
        return samples
    
    def run_all_validations(self, samples: List[ValidationSample]):
        """运行所有验证"""
        print("="*70)
        print("Stage 11-A-R2-Phase2.5: 破满分验证")
        print("="*70)
        print(f"总样本数: {len(samples)}")
        print("="*70)
        
        # 1. 模板隔离切分验证
        print("\n" + "="*70)
        print("[验证1/5] 模板隔离切分验证")
        print("="*70)
        self._validate_template_isolation(samples)
        
        # 2. 去捷径字段消融
        print("\n" + "="*70)
        print("[验证2/5] 去捷径字段消融实验")
        print("="*70)
        self._validate_field_ablation(samples)
        
        # 3. 对抗样本测试
        print("\n" + "="*70)
        print("[验证3/5] 对抗样本测试")
        print("="*70)
        self._validate_adversarial_samples()
        
        # 4. 标签独立性检查
        print("\n" + "="*70)
        print("[验证4/5] 标签独立性检查")
        print("="*70)
        self._validate_label_independence(samples)
        
        # 5. 人工盲审准备
        print("\n" + "="*70)
        print("[验证5/5] 人工盲审样本准备")
        print("="*70)
        self._prepare_human_review(samples)
        
        # 总结
        self._print_summary()
    
    def _validate_template_isolation(self, samples: List[ValidationSample]):
        """模板隔离切分验证"""
        print("\n  按不同维度切分训练和验证集...")
        
        # 按样本类型切分
        type_groups = defaultdict(list)
        for s in samples:
            type_groups[s.sample_type].append(s)
        
        print("\n  【按样本类型隔离验证】")
        for sample_type, group_samples in type_groups.items():
            if len(group_samples) < 5:
                continue
            
            # 用这个类型作为验证集，其他作为训练集
            train_samples = [s for s in samples if s.sample_type != sample_type]
            val_samples = group_samples
            
            # 评估验证集
            correct = {'gap': 0, 'retrieval': 0, 'tsla': 0, 'memory': 0}
            total = {'gap': 0, 'retrieval': 0, 'tsla': 0, 'memory': 0}
            
            for s in val_samples:
                result = self.validator.evaluate_sample(s)
                for key in correct.keys():
                    if f'{key}_correct' in result:
                        correct[key] += result[f'{key}_correct']
                        total[key] += 1
            
            print(f"\n    验证类型: {sample_type} ({len(val_samples)}条)")
            for key in ['gap', 'retrieval', 'tsla', 'memory']:
                if total[key] > 0:
                    acc = correct[key] / total[key]
                    status = "✓" if acc >= 0.85 else "⚠"
                    print(f"      {status} {key}: {acc:.1%}")
        
        # 按场景类型切分
        print("\n  【按场景类型隔离验证】")
        scenario_groups = defaultdict(list)
        for s in samples:
            scenario_groups[s.scenario_type].append(s)
        
        for scenario_type, group_samples in scenario_groups.items():
            if len(group_samples) < 5:
                continue
            
            val_samples = group_samples
            correct = {'gap': 0, 'retrieval': 0}
            
            for s in val_samples:
                result = self.validator.evaluate_sample(s)
                correct['gap'] += result.get('gap_correct', 0)
                correct['retrieval'] += result.get('retrieval_correct', 0)
            
            gap_acc = correct['gap'] / len(val_samples)
            ret_acc = correct['retrieval'] / len(val_samples)
            print(f"    {scenario_type}: gap={gap_acc:.1%}, retrieval={ret_acc:.1%}")
        
        # 按难度切分
        print("\n  【按难度段隔离验证】")
        for difficulty in [1, 2, 3, 4, 5]:
            diff_samples = [s for s in samples if s.difficulty == difficulty]
            if len(diff_samples) < 5:
                continue
            
            correct = {'gap': 0, 'retrieval': 0}
            for s in diff_samples:
                result = self.validator.evaluate_sample(s)
                correct['gap'] += result.get('gap_correct', 0)
                correct['retrieval'] += result.get('retrieval_correct', 0)
            
            gap_acc = correct['gap'] / len(diff_samples)
            ret_acc = correct['retrieval'] / len(diff_samples)
            print(f"    难度{difficulty}: gap={gap_acc:.1%}, retrieval={ret_acc:.1%} ({len(diff_samples)}条)")
    
    def _validate_field_ablation(self, samples: List[ValidationSample]):
        """去捷径字段消融实验"""
        print("\n  测试去掉不同字段后的性能变化...")
        
        # 完整输入 (baseline)
        print("\n  【Baseline: 完整输入】")
        baseline = self._evaluate_with_config(samples, True, True)
        print(f"    gap: {baseline['gap']:.1%}, retrieval: {baseline['retrieval']:.1%}")
        
        # 去掉 known_info
        print("\n  【消融1: 去掉 known_info】")
        no_known = self._evaluate_with_config(samples, False, True)
        print(f"    gap: {no_known['gap']:.1%} (↓{baseline['gap']-no_known['gap']:.1%})")
        print(f"    retrieval: {no_known['retrieval']:.1%} (↓{baseline['retrieval']-no_known['retrieval']:.1%})")
        
        if baseline['gap'] - no_known['gap'] > 0.2:
            print("    ⚠️ 严重依赖 known_info，存在捷径学习！")
        
        # 去掉 scenario_type
        print("\n  【消融2: 去掉 scenario_type】")
        no_scenario = self._evaluate_with_config(samples, True, False)
        print(f"    gap: {no_scenario['gap']:.1%} (↓{baseline['gap']-no_scenario['gap']:.1%})")
        
        # 全部去掉
        print("\n  【消融3: 只保留 query】")
        minimal = self._evaluate_with_config(samples, False, False)
        print(f"    gap: {minimal['gap']:.1%} (↓{baseline['gap']-minimal['gap']:.1%})")
        print(f"    retrieval: {minimal['retrieval']:.1%} (↓{baseline['retrieval']-minimal['retrieval']:.1%})")
        
        if baseline['gap'] - minimal['gap'] > 0.3:
            print("    ⚠️ 严重依赖辅助字段，真实泛化能力存疑！")
    
    def _evaluate_with_config(self, samples: List[ValidationSample], 
                              use_known_info: bool, use_scenario: bool) -> Dict:
        """使用指定配置评估"""
        correct = {'gap': 0, 'retrieval': 0}
        for s in samples:
            result = self.validator.evaluate_sample(s, use_known_info, use_scenario)
            correct['gap'] += result.get('gap_correct', 0)
            correct['retrieval'] += result.get('retrieval_correct', 0)
        
        return {
            'gap': correct['gap'] / len(samples),
            'retrieval': correct['retrieval'] / len(samples),
        }
    
    def _validate_adversarial_samples(self):
        """对抗样本测试"""
        print("\n  生成对抗样本并测试...")
        
        adversarial_samples = [
            # 表面完整、实则缺关键事实
            ValidationSample(
                id="adv_001",
                query="根据最新政策，我们应该如何调整项目计划？",
                known_info="提到了最新政策，但未提供具体政策内容",
                scenario_type="adversarial",
                sample_type="缺口识别型",
                sample_family="边界模糊型",
                difficulty=4,
                model_targets={'gap_detected': 1, 'retrieval_needed': 1},
            ),
            # 表面像要检索，其实闭卷足够
            ValidationSample(
                id="adv_002",
                query="请解释什么是机器学习中的过拟合现象？",
                known_info="这是一个标准概念解释问题",
                scenario_type="adversarial",
                sample_type="教师引导型",
                sample_family="明确无需检索型",
                difficulty=3,
                model_targets={'gap_detected': 0, 'retrieval_needed': 0},
            ),
            # 用户错误前提 + 高流畅表达
            ValidationSample(
                id="adv_003",
                query="既然地球是平的，那么航海时需要注意什么？",
                known_info="问题基于错误前提(地球是平的)",
                scenario_type="adversarial",
                sample_type="缺口识别型",
                sample_family="用户错误前提型",
                difficulty=5,
                model_targets={'gap_detected': 1, 'retrieval_needed': 1},
            ),
            # 多来源轻微冲突
            ValidationSample(
                id="adv_004",
                query="根据A专家和B专家的不同观点，这个技术方案可行吗？",
                known_info="存在轻微冲突的专家意见",
                scenario_type="adversarial",
                sample_type="检索整合型",
                sample_family="冲突/多解型",
                difficulty=4,
                model_targets={'gap_detected': 1, 'retrieval_needed': 1},
            ),
            # 需要拆分但不明显的混义样本
            ValidationSample(
                id="adv_005",
                query="这个方案怎么样？",
                known_info="问题过于模糊，可能指多个不同方案",
                scenario_type="adversarial",
                sample_type="缺口识别型",
                sample_family="边界模糊型",
                difficulty=5,
                model_targets={'gap_detected': 1, 'retrieval_needed': 1},
            ),
        ]
        
        print("\n  【对抗样本表现】")
        for sample in adversarial_samples:
            result = self.validator.evaluate_sample(sample)
            gap_ok = "✓" if result.get('gap_correct') else "✗"
            ret_ok = "✓" if result.get('retrieval_correct') else "✗"
            print(f"    {sample.id} (难度{sample.difficulty}): gap{gap_ok} retrieval{ret_ok}")
            print(f"      问题: {sample.query[:40]}...")
        
        # 统计
        gap_correct = sum(1 for s in adversarial_samples 
                         if self.validator.evaluate_sample(s).get('gap_correct'))
        ret_correct = sum(1 for s in adversarial_samples 
                         if self.validator.evaluate_sample(s).get('retrieval_correct'))
        
        print(f"\n  对抗样本准确率: gap={gap_correct/len(adversarial_samples):.0%}, "
              f"retrieval={ret_correct/len(adversarial_samples):.0%}")
        
        if gap_correct < len(adversarial_samples):
            print("  ⚠️ 对抗样本存在失败，泛化能力需要加强！")
    
    def _validate_label_independence(self, samples: List[ValidationSample]):
        """标签独立性检查"""
        print("\n  检查TSLA/Memory标签是否独立于上游决策...")
        
        # 统计不同gap值下的TSLA分布
        tsla_by_gap = defaultdict(lambda: defaultdict(int))
        memory_by_retrieval = defaultdict(lambda: defaultdict(int))
        
        for s in samples:
            gap = s.model_targets.get('gap_detected', 0)
            retrieval = s.model_targets.get('retrieval_needed', 0)
            
            if 'tsla_action' in s.model_targets:
                tsla_by_gap[gap][s.model_targets['tsla_action']] += 1
            
            if 'memory_action' in s.model_targets:
                memory_by_retrieval[retrieval][s.model_targets['memory_action']] += 1
        
        print("\n  【TSLA动作 vs Gap决策】")
        for gap_val in [0, 1]:
            print(f"\n    Gap={gap_val}时的TSLA分布:")
            total = sum(tsla_by_gap[gap_val].values())
            if total > 0:
                for action, count in tsla_by_gap[gap_val].items():
                    print(f"      {action}: {count} ({count/total:.1%})")
        
        # 检查是否gap=0和gap=1时TSLA分布差异大
        gap0_actions = set(tsla_by_gap[0].keys())
        gap1_actions = set(tsla_by_gap[1].keys())
        
        if gap0_actions == gap1_actions and len(gap0_actions) > 1:
            print("\n  ✓ TSLA动作在gap=0和gap=1时都有多种选择，显示一定独立性")
        else:
            print("\n  ⚠️ TSLA动作和gap决策高度耦合，可能不是独立学习！")
        
        print("\n  【Memory动作 vs Retrieval决策】")
        for ret_val in [0, 1]:
            print(f"\n    Retrieval={ret_val}时的Memory分布:")
            total = sum(memory_by_retrieval[ret_val].values())
            if total > 0:
                for action, count in memory_by_retrieval[ret_val].items():
                    print(f"      {action}: {count} ({count/total:.1%})")
    
    def _prepare_human_review(self, samples: List[ValidationSample]):
        """准备人工盲审样本"""
        print("\n  抽取50条样本用于人工盲审...")
        
        # 按类型和难度分层抽样
        review_samples = []
        
        # 从每种类型抽10条
        type_groups = defaultdict(list)
        for s in samples:
            type_groups[s.sample_type].append(s)
        
        for sample_type, group in type_groups.items():
            # 优先选择难度高的
            group.sort(key=lambda x: x.difficulty, reverse=True)
            selected = group[:10]
            review_samples.extend(selected)
        
        # 打乱顺序
        random.shuffle(review_samples)
        review_samples = review_samples[:50]
        
        print(f"\n  已抽取 {len(review_samples)} 条样本")
        
        # 生成盲审表格
        review_data = []
        for i, s in enumerate(review_samples, 1):
            review_data.append({
                '序号': i,
                '样本ID': s.id,
                '问题': s.query,
                '已知信息': s.known_info,
                '难度': s.difficulty,
                '模型预测': '待评估',
                '人工判断': '',
                '是否一致': '',
            })
        
        # 保存
        review_path = 'stage8_dataset/human_review_50_samples.json'
        with open(review_path, 'w', encoding='utf-8') as f:
            json.dump(review_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n  ✓ 盲审表格已保存: {review_path}")
        print("\n  盲审说明:")
        print("    1. 只看'问题'和'已知信息'两列")
        print("    2. 人工判断: gap=0/1, retrieval=0/1")
        print("    3. 对比模型预测，标记是否一致")
        print("    4. 重点关注难度≥3的样本")
    
    def _print_summary(self):
        """打印验证总结"""
        print("\n" + "="*70)
        print("Phase2.5 验证总结")
        print("="*70)
        print("\n关键发现:")
        print("  1. 如果模板隔离后准确率下降 → 存在模板记忆")
        print("  2. 如果去掉known_info后准确率下降>20% → 捷径学习")
        print("  3. 如果对抗样本失败 → 泛化能力不足")
        print("  4. 如果TSLA和gap高度耦合 → 不是独立能力")
        print("\n下一步建议:")
        print("  - 完成人工盲审50条")
        print("  - 根据验证结果决定是否需要:")
        print("    a) 增加对抗样本训练")
        print("    b) 减少辅助字段依赖")
        print("    c) 解耦标签设计")
        print("    d) 进入Stage 11-B")
        print("="*70)


def run_phase25_validation():
    """运行Phase2.5验证"""
    print("="*70)
    print("Stage 11-A-R2-Phase2.5: 破满分验证启动")
    print("="*70)
    
    # 加载模型
    print("\n[准备] 加载Phase2训练好的模型...")
    config = NativeTinyConfig()
    base_model = NativeBackboneTinyV1(config)
    
    # 这里需要导入Phase2的模型类
    from stage11a_r2_phase2_trainer import Phase2SelfLearningModel
    model = Phase2SelfLearningModel(base_model)
    
    checkpoint = torch.load('stage8_dataset/stage11a_r2_phase2_checkpoint.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    print("  ✓ 模型加载完成")
    
    # 加载样本
    print("\n[准备] 加载Phase2数据集...")
    validation_suite = Phase25ValidationSuite(model)
    samples = validation_suite.load_samples('stage8_dataset/stage11a_r2_phase2_dataset.json')
    print(f"  ✓ 加载 {len(samples)} 条样本")
    
    # 运行验证
    print("\n[执行] 运行完整验证套件...")
    validation_suite.run_all_validations(samples)
    
    return validation_suite


if __name__ == "__main__":
    suite = run_phase25_validation()
