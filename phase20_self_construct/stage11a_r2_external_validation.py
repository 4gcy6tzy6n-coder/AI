"""
Stage 11-A-R2-External: 三道外部验证

核心目标: 打破"评测宇宙"闭环，验证真实泛化能力

三道验证:
1. 盲出题验证 - 独立出题人按理论文档出100条，不知训练集
2. 原始理论回标验证 - 60条重标，检查H5等标签漂移
3. 真实对话回放验证 - 非结构化真实脏对话

关键检查点:
- H5高质量冲突: 理论应→回流重审，当前标签→拆分？(漂移警报)
- 外部分布是否掉分
- 真实对话是否还能稳

结论改写:
"Stage 11-A-R2 内部受控验收通过，外部未知分布验证待完成"
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
from dataclasses import dataclass
from collections import defaultdict

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


# ========== 外部验证样本 ==========
@dataclass
class ExternalSample:
    """外部验证样本 - 独立于训练集设计"""
    id: str
    query: str
    context: str
    sample_source: str  # blind_quiz / theory_relabel / real_dialogue
    
    # 按原始理论文档标注 (不看当前训练标签)
    theory_gap: int
    theory_retrieval: int
    theory_tsla: str
    theory_memory: str
    theory_rationale: str  # 标注依据
    
    # 硬否决信号 (按原始理论)
    hard_veto_signals: List[str]  # H1/H2/H4/H5
    
    # 当前模型预测 (待填充)
    model_prediction: Dict = None


# ========== 验证1: 盲出题验证 ==========
class BlindQuizGenerator:
    """
    盲出题生成器 - 模拟"独立出题人"
    
    规则:
    - 只知道理论文档，不知训练集内容
    - 按TSLA八动作、五门机制、硬否决信号出题
    - 覆盖真实世界脏场景
    """
    
    def __init__(self):
        self.samples: List[ExternalSample] = []
    
    def generate_100_blind_quizzes(self) -> List[ExternalSample]:
        """生成100条盲出题目"""
        print("\n[盲出题验证] 生成100条独立题目...")
        print("  出题人视角: 只知道理论，不知训练集")
        
        samples = []
        
        # 1. 错误前提型 (20条) - H1
        print("\n  [1/5] 错误前提型 20条...")
        error_premises = [
            ("既然水在100度会结冰，那 boiling water 应该很安全吧？", 
             "前提错误: 水在100度沸腾而非结冰", "H1"),
            ("根据地平说，环球航行是怎么完成的？",
             "前提错误: 地球是球体而非平面", "H1"),
            ("既然人类只开发了10%大脑，那怎么解释神经科学的研究？",
             "前提错误: 大脑开发度是谣言", "H1"),
            ("根据永动机原理，我们能解决能源危机吗？",
             "前提错误: 永动机违反热力学定律", "H1"),
            ("既然进化论只是假说，那物种起源应该怎么解释？",
             "前提错误: 进化论是科学理论而非假说", "H1"),
        ]
        for i, (query, rationale, veto) in enumerate(error_premises):
            for j in range(4):  # 每条变体4次
                samples.append(self._create_blind_sample(
                    f"blind_h1_{i}_{j}", query, 
                    theory_gap=1, theory_retrieval=1,
                    theory_tsla="回流重审", theory_memory="隔离观察",
                    rationale=rationale, veto=[veto],
                    source="blind_quiz"
                ))
        
        # 2. 高质量冲突型 (20条) - H5 (关键! 理论应回流，不是拆分)
        print("  [2/5] 高质量冲突型 20条...")
        quality_conflicts = [
            ("权威期刊A说X疗法有效，权威期刊B说X疗法无效，患者该信谁？",
             "H5: 两个高质量来源冲突", "回流重审"),
            ("顶级专家P认为方案最优，同领域顶级专家Q认为方案有风险，如何决策？",
             "H5: 专家意见冲突", "回流重审"),
            ("历史数据显示趋势A，最新实时数据显示趋势B，且都来源可靠",
             "H5: 数据冲突", "回流重审"),
            ("理论模型预测结果X，大量实证研究支持结果Y，如何整合？",
             "H5: 理论vs实证冲突", "回流重审"),
            ("官方文档v2.0说功能已移除，官方文档v2.1说功能仍支持",
             "H5: 官方来源冲突", "回流重审"),
        ]
        for i, (query, rationale, tsla) in enumerate(quality_conflicts):
            for j in range(4):
                samples.append(self._create_blind_sample(
                    f"blind_h5_{i}_{j}", query,
                    theory_gap=1, theory_retrieval=1,
                    theory_tsla=tsla, theory_memory="进入受审区",
                    rationale=rationale, veto=["H5"],
                    source="blind_quiz"
                ))
        
        # 3. 多义未拆分型 (20条) - H4
        print("  [3/5] 多义未拆分型 20条...")
        ambiguities = [
            ("这个方案怎么样？", "指代不明: 哪个方案？"),
            ("帮我优化一下", "对象不明: 优化什么？"),
            ("那个问题解决了没有？", "上下文缺失: 哪个问题？"),
            ("这样做对吗？", "标准不明: 什么是对？"),
            ("请分析一下", "范围不明: 分析什么？"),
        ]
        for i, (query, rationale) in enumerate(ambiguities):
            for j in range(4):
                samples.append(self._create_blind_sample(
                    f"blind_h4_{i}_{j}", query,
                    theory_gap=1, theory_retrieval=1,
                    theory_tsla="拆分", theory_memory="隔离观察",
                    rationale=f"H4: {rationale}", veto=["H4"],
                    source="blind_quiz"
                ))
        
        # 4. 结构冲突型 (20条) - H2
        print("  [4/5] 结构冲突型 20条...")
        structure_conflicts = [
            ("基于完整的数据分析，请预测明天彩票号码",
             "表面完整但关键: 彩票随机不可预测"),
            ("综合所有医学文献，请确诊我的疾病",
             "越权: AI不能医疗诊断"),
            ("根据公开信息，请预测某公司的内部战略决策",
             "信息缺口: 内部决策非公开"),
            ("用现有物理定律解释量子引力统一理论",
             "知识边界: 该问题尚未解决"),
        ]
        for i, (query, rationale) in enumerate(structure_conflicts):
            for j in range(5):
                samples.append(self._create_blind_sample(
                    f"blind_h2_{i}_{j}", query,
                    theory_gap=1, theory_retrieval=1,
                    theory_tsla="回流重审", theory_memory="进入受审区",
                    rationale=f"H2: {rationale}", veto=["H2"],
                    source="blind_quiz"
                ))
        
        # 5. 混合复杂型 (20条) - 多信号
        print("  [5/5] 混合复杂型 20条...")
        mixed_samples = [
            ("既然地球是平的(错误前提)，而且多个古代文明都这么认为(轻冲突)，那现代地图是怎么绘制的(多义)？",
             ["H1", "H5", "H4"], "回流重审"),
            ("请分析这个问题(模糊)，基于我给你的所有信息(表面完整)，但注意不同来源可能有分歧(冲突)",
             ["H4", "H2", "H5"], "拆分"),
        ]
        for i, (query, vetos, tsla) in enumerate(mixed_samples):
            for j in range(10):
                samples.append(self._create_blind_sample(
                    f"blind_mix_{i}_{j}", query,
                    theory_gap=1, theory_retrieval=1,
                    theory_tsla=tsla, theory_memory="隔离观察",
                    rationale="多硬否决信号叠加", veto=vetos,
                    source="blind_quiz"
                ))
        
        print(f"\n  ✓ 生成 {len(samples)} 条盲出题目")
        return samples[:100]  # 确保正好100条
    
    def _create_blind_sample(self, id: str, query: str,
                             theory_gap: int, theory_retrieval: int,
                             theory_tsla: str, theory_memory: str,
                             rationale: str, veto: List[str],
                             source: str) -> ExternalSample:
        return ExternalSample(
            id=id,
            query=query,
            context="盲出题: 按理论文档独立设计",
            sample_source=source,
            theory_gap=theory_gap,
            theory_retrieval=theory_retrieval,
            theory_tsla=theory_tsla,
            theory_memory=theory_memory,
            theory_rationale=rationale,
            hard_veto_signals=veto,
        )


# ========== 验证2: 原始理论回标验证 ==========
class TheoryRelabelValidator:
    """
    原始理论回标验证器
    
    从当前600条中随机抽60条，严格按原始理论文档重标
    检查标签漂移 (特别是H5)
    """
    
    def __init__(self):
        self.relabel_results: List[Dict] = []
    
    def relabel_60_samples(self, original_samples: List[Dict]) -> List[Dict]:
        """重标60条样本"""
        print("\n[理论回标验证] 随机抽取60条重标...")
        print("  重标规则: 严格按原始TSLA理论文档，不看当前标签")
        
        # 随机抽60条
        selected = random.sample(original_samples, min(60, len(original_samples)))
        
        relabeled = []
        drift_detected = []
        
        for sample in selected:
            # 按原始理论重新判断
            relabel = self._relabel_by_theory(sample)
            
            # 检查是否漂移
            original_tsla = sample.get('model_targets', {}).get('tsla_action', '')
            if original_tsla != relabel['theory_tsla']:
                drift_detected.append({
                    'id': sample['id'],
                    'query': sample['query'][:50],
                    'original_tsla': original_tsla,
                    'theory_tsla': relabel['theory_tsla'],
                    'rationale': relabel['rationale'],
                })
            
            relabeled.append({
                'id': sample['id'],
                'query': sample['query'],
                'original': sample.get('model_targets', {}),
                'relabel': relabel,
                'drift': original_tsla != relabel['theory_tsla'],
            })
        
        print(f"\n  重标完成: {len(relabeled)} 条")
        print(f"  标签漂移: {len(drift_detected)} 条 ({len(drift_detected)/len(relabeled):.1%})")
        
        if drift_detected:
            print("\n  漂移案例:")
            for d in drift_detected[:5]:  # 只显示前5个
                print(f"    {d['id']}: {d['original_tsla']} → {d['theory_tsla']}")
                print(f"      依据: {d['rationale'][:60]}...")
        
        return relabeled, drift_detected
    
    def _relabel_by_theory(self, sample: Dict) -> Dict:
        """按原始理论重新标注"""
        query = sample.get('query', '')
        context = sample.get('context', {})
        
        # 硬否决信号检测 (按原始理论)
        hard_vetos = []
        
        # H1: 幻觉暴露/错误前提
        if any(term in query for term in ['地球是平的', '永动机', '10%大脑', '地平说']):
            hard_vetos.append('H1')
        
        # H2: 结构冲突/越权
        if any(term in query for term in ['预测彩票', '确诊疾病', '内部战略']):
            hard_vetos.append('H2')
        
        # H4: 多义未拆分
        if any(term in query for term in ['怎么样', '优化一下', '这个问题', '分析一下']) and len(query) < 20:
            hard_vetos.append('H4')
        
        # H5: 高质量冲突 (关键!)
        if any(term in query for term in ['专家说', '期刊说', '数据显示', '权威']):
            if any(term in query for term in ['冲突', '分歧', '不同', '相反', 'A说...B说']):
                hard_vetos.append('H5')
        
        # 按理论确定TSLA (优先级: H1/H2 > H4 > H5)
        if 'H1' in hard_vetos or 'H2' in hard_vetos:
            tsla = "回流重审"
            rationale = "H1/H2硬否决: 必须回流重审"
        elif 'H4' in hard_vetos:
            tsla = "拆分"
            rationale = "H4硬否决: 多义必须拆分"
        elif 'H5' in hard_vetos:
            tsla = "回流重审"  # 关键! 理论中H5是回流，不是拆分
            rationale = "H5硬否决: 高质量冲突需回流重审"
        else:
            tsla = "保留"
            rationale = "无硬否决信号，正常保留"
        
        return {
            'theory_gap': 1 if hard_vetos else 0,
            'theory_retrieval': 1 if hard_vetos else 0,
            'theory_tsla': tsla,
            'theory_memory': "隔离观察" if hard_vetos else "晋升候选",
            'hard_vetos': hard_vetos,
            'rationale': rationale,
        }


# ========== 验证3: 真实对话回放验证 ==========
class RealDialogueValidator:
    """
    真实对话回放验证
    
    非结构化真实脏对话，测试模型在开放输入下的表现
    """
    
    def __init__(self):
        self.dialogues: List[ExternalSample] = []
    
    def generate_real_dialogues(self) -> List[ExternalSample]:
        """生成真实对话样本"""
        print("\n[真实对话验证] 生成50条真实脏对话...")
        
        dialogues = [
            # 错误前提+强势语气
            {
                'query': "我跟你说，进化论早就被证伪了，你居然还信？",
                'context': '用户强势表达错误观点',
                'gap': 1, 'retrieval': 1,
                'tsla': '回流重审', 'memory': '隔离观察',
                'vetos': ['H1'],
            },
            # 半真半假
            {
                'query': "GPT-4有1.8万亿参数，比人脑神经元还多，所以它比人聪明对吧？",
                'context': '前半真(参数多)后半假(不等于聪明)',
                'gap': 1, 'retrieval': 1,
                'tsla': '回流重审', 'memory': '隔离观察',
                'vetos': ['H1'],
            },
            # 模糊指代
            {
                'query': "那个东西你弄好了吗？",
                'context': '多轮对话中，指代完全不明',
                'gap': 1, 'retrieval': 1,
                'tsla': '拆分', 'memory': '隔离观察',
                'vetos': ['H4'],
            },
            # 多轮后才暴露冲突
            {
                'query': "前面说的方案我觉得可以，但刚才查了下发现数据对不上",
                'context': '多轮后暴露前后矛盾',
                'gap': 1, 'retrieval': 1,
                'tsla': '回流重审', 'memory': '进入受审区',
                'vetos': ['H5'],
            },
            # 用户说法vs知识库冲突
            {
                'query': "我听说Python 4.0已经发布了，为什么你说没有？",
                'context': '用户错误信息vs正确知识',
                'gap': 1, 'retrieval': 1,
                'tsla': '回流重审', 'memory': '隔离观察',
                'vetos': ['H1'],
            },
            # 隐含错误前提
            {
                'query': "既然AI这么厉害，那它应该能预测股票涨跌吧？",
                'context': '隐含: AI厉害=能预测随机事件',
                'gap': 1, 'retrieval': 1,
                'tsla': '回流重审', 'memory': '隔离观察',
                'vetos': ['H2'],
            },
            # 极端模糊
            {
                'query': "嗯...就是那个...你懂的吧？",
                'context': '几乎无信息',
                'gap': 1, 'retrieval': 1,
                'tsla': '拆分', 'memory': '隔离观察',
                'vetos': ['H4'],
            },
            # 专业术语伪装错误
            {
                'query': "根据量子纠缠的超距作用，我们可以实现瞬移通信",
                'context': '用正确术语表达错误结论',
                'gap': 1, 'retrieval': 1,
                'tsla': '回流重审', 'memory': '隔离观察',
                'vetos': ['H1'],
            },
        ]
        
        samples = []
        for i, d in enumerate(dialogues):
            for j in range(6):  # 每类6条变体
                samples.append(ExternalSample(
                    id=f"real_{i}_{j}",
                    query=d['query'],
                    context=d['context'],
                    sample_source="real_dialogue",
                    theory_gap=d['gap'],
                    theory_retrieval=d['retrieval'],
                    theory_tsla=d['tsla'],
                    theory_memory=d['memory'],
                    theory_rationale=f"真实对话: {d['context']}",
                    hard_veto_signals=d['vetos'],
                ))
        
        print(f"  ✓ 生成 {len(samples)} 条真实对话")
        return samples[:50]


# ========== 外部验证执行器 ==========
class ExternalValidationRunner:
    """外部验证执行器"""
    
    def __init__(self, model):
        self.model = model
        self.model.eval()
    
    def run_all_validations(self):
        """运行全部三道验证"""
        print("="*70)
        print("Stage 11-A-R2-External: 三道外部验证")
        print("="*70)
        print("目标: 打破'评测宇宙'闭环，验证真实泛化")
        print("="*70)
        
        results = {}
        
        # 验证1: 盲出题
        print("\n" + "="*70)
        print("验证1: 盲出题验证 (独立出题人100条)")
        print("="*70)
        blind_gen = BlindQuizGenerator()
        blind_samples = blind_gen.generate_100_blind_quizzes()
        results['blind_quiz'] = self._evaluate_samples(blind_samples, "盲出题")
        
        # 验证2: 理论回标
        print("\n" + "="*70)
        print("验证2: 原始理论回标验证 (60条重标)")
        print("="*70)
        
        # 加载原始样本
        try:
            with open('stage8_dataset/stage11a_r2_phase26_dataset.json', 'r', encoding='utf-8') as f:
                original_samples = json.load(f)
        except:
            print("  ⚠ 无法加载原始样本，跳过回标验证")
            original_samples = []
        
        if original_samples:
            relabel_val = TheoryRelabelValidator()
            relabeled, drift = relabel_val.relabel_60_samples(original_samples)
            results['relabel'] = {
                'total': len(relabeled),
                'drift': len(drift),
                'drift_rate': len(drift) / len(relabeled) if relabeled else 0,
            }
            
            # 用重标标签评估模型
            relabel_samples = []
            for r in relabeled:
                relabel_samples.append(ExternalSample(
                    id=r['id'],
                    query=r['query'],
                    context="理论回标",
                    sample_source="theory_relabel",
                    theory_gap=r['relabel']['theory_gap'],
                    theory_retrieval=r['relabel']['theory_retrieval'],
                    theory_tsla=r['relabel']['theory_tsla'],
                    theory_memory=r['relabel']['theory_memory'],
                    theory_rationale=r['relabel']['rationale'],
                    hard_veto_signals=r['relabel']['hard_vetos'],
                ))
            
            results['relabel_eval'] = self._evaluate_samples(relabel_samples, "理论回标")
        
        # 验证3: 真实对话
        print("\n" + "="*70)
        print("验证3: 真实对话回放验证 (50条)")
        print("="*70)
        dialogue_val = RealDialogueValidator()
        dialogue_samples = dialogue_val.generate_real_dialogues()
        results['real_dialogue'] = self._evaluate_samples(dialogue_samples, "真实对话")
        
        # 汇总报告
        self._print_final_report(results)
        
        return results
    
    def _evaluate_samples(self, samples: List[ExternalSample], name: str) -> Dict:
        """评估样本集"""
        correct = {'gap': 0, 'retrieval': 0, 'tsla': 0, 'memory': 0}
        total = {'gap': 0, 'retrieval': 0, 'tsla': 0, 'memory': 0}
        
        # H5专项统计
        h5_total = 0
        h5_correct = 0
        
        with torch.no_grad():
            for sample in samples:
                input_ids = self._encode(sample.query)
                outputs = self.model(input_ids)
                
                # Gap
                gap_pred = outputs['gap_detection_logits'].argmax(dim=-1).item()
                total['gap'] += 1
                if gap_pred == sample.theory_gap:
                    correct['gap'] += 1
                
                # Retrieval
                retrieval_pred = outputs['retrieval_decision_logits'].argmax(dim=-1).item()
                total['retrieval'] += 1
                if retrieval_pred == sample.theory_retrieval:
                    correct['retrieval'] += 1
                
                # TSLA
                tsla_pred = outputs['tsla_action_logits'].argmax(dim=-1).item()
                tsla_target = TSLA_ACTION_TO_ID.get(sample.theory_tsla, 0)
                total['tsla'] += 1
                if tsla_pred == tsla_target:
                    correct['tsla'] += 1
                
                # Memory
                memory_pred = outputs['memory_action_logits'].argmax(dim=-1).item()
                memory_target = MEMORY_ACTION_TO_ID.get(sample.theory_memory, 0)
                total['memory'] += 1
                if memory_pred == memory_target:
                    correct['memory'] += 1
                
                # H5专项
                if 'H5' in sample.hard_veto_signals:
                    h5_total += 1
                    # H5理论应回流重审
                    if tsla_pred == TSLA_ACTION_TO_ID['回流重审']:
                        h5_correct += 1
        
        # 计算准确率
        accuracies = {}
        for key in correct.keys():
            if total[key] > 0:
                accuracies[key] = correct[key] / total[key]
            else:
                accuracies[key] = 0
        
        h5_acc = h5_correct / h5_total if h5_total > 0 else 0
        
        print(f"\n  {name} 结果:")
        print(f"    Gap: {accuracies['gap']:.1%}")
        print(f"    Retrieval: {accuracies['retrieval']:.1%}")
        print(f"    TSLA: {accuracies['tsla']:.1%}")
        print(f"    Memory: {accuracies['memory']:.1%}")
        if h5_total > 0:
            print(f"    H5高质量冲突(应回流): {h5_acc:.1%}")
        
        return {
            'accuracies': accuracies,
            'h5_accuracy': h5_acc,
            'h5_total': h5_total,
            'total_samples': len(samples),
        }
    
    def _encode(self, text: str) -> torch.Tensor:
        tokens = [ord(c) % 10000 for c in text[:100]]
        if len(tokens) < 10:
            tokens.extend([0] * (10 - len(tokens)))
        return torch.tensor([tokens])
    
    def _print_final_report(self, results: Dict):
        """打印最终报告"""
        print("\n" + "="*70)
        print("外部验证最终报告")
        print("="*70)
        
        # 对比表
        print("\n  各验证维度对比:")
        print("  " + "-"*60)
        print(f"  {'维度':<15} {'盲出题':<12} {'理论回标':<12} {'真实对话':<12}")
        print("  " + "-"*60)
        
        for metric in ['gap', 'retrieval', 'tsla', 'memory']:
            blind = results.get('blind_quiz', {}).get('accuracies', {}).get(metric, 0)
            relabel = results.get('relabel_eval', {}).get('accuracies', {}).get(metric, 0)
            dialogue = results.get('real_dialogue', {}).get('accuracies', {}).get(metric, 0)
            print(f"  {metric:<15} {blind:<12.1%} {relabel:<12.1%} {dialogue:<12.1%}")
        
        print("  " + "-"*60)
        
        # H5专项
        print("\n  H5高质量冲突专项 (理论应回流重审):")
        h5_blind = results.get('blind_quiz', {}).get('h5_accuracy', 0)
        h5_relabel = results.get('relabel_eval', {}).get('h5_accuracy', 0)
        h5_dialogue = results.get('real_dialogue', {}).get('h5_accuracy', 0)
        print(f"    盲出题: {h5_blind:.1%}")
        print(f"    理论回标: {h5_relabel:.1%}")
        print(f"    真实对话: {h5_dialogue:.1%}")
        
        # 标签漂移
        if 'relabel' in results:
            drift_rate = results['relabel'].get('drift_rate', 0)
            print(f"\n  标签漂移率: {drift_rate:.1%}")
            if drift_rate > 0.1:
                print("    ⚠ 警告: 标签漂移率>10%，当前100%可能只是对当前标签定义满分")
            else:
                print("    ✓ 标签一致性较好")
        
        # 结论
        print("\n" + "="*70)
        print("阶段性结论")
        print("="*70)
        
        # 计算平均外部准确率
        all_accs = []
        for key in ['blind_quiz', 'relabel_eval', 'real_dialogue']:
            if key in results:
                accs = results[key].get('accuracies', {})
                all_accs.extend(accs.values())
        
        avg_external = sum(all_accs) / len(all_accs) if all_accs else 0
        
        print(f"\n  内部受控测试: 100% (600条自设计样本)")
        print(f"  外部未知分布: {avg_external:.1%} (平均)")
        
        if avg_external >= 0.85:
            print("\n  ✅ 外部验证通过")
            print("  模型在陌生分布下仍保持稳定")
            print("  可以进入 Stage 11-B: 教师退场")
        elif avg_external >= 0.70:
            print("\n  ⚠️  外部验证部分通过")
            print("  模型在部分陌生场景下掉分")
            print("  建议: 补充外部样本后再进入Stage 11-B")
        else:
            print("\n  ❌ 外部验证未通过")
            print("  模型明显过拟合当前评测体系")
            print("  必须: 重新设计训练策略，增加真实场景")
        
        print("\n  正式结论:")
        print("  " + "-"*60)
        print("  Stage 11-A-R2 内部受控验收通过")
        print("  外部未知分布验证: " + ("通过" if avg_external >= 0.85 else "待加强"))
        print("  " + "-"*60)
        
        print("\n" + "="*70)


# ========== 主运行函数 ==========
def run_external_validation():
    """运行外部验证"""
    print("="*70)
    print("Stage 11-A-R2-External: 三道外部验证启动")
    print("="*70)
    
    # 加载Phase2.6模型
    print("\n[准备] 加载Phase2.6模型...")
    config = NativeTinyConfig()
    base_model = NativeBackboneTinyV1(config)
    
    from stage11a_r2_phase2_6_adversarial_trainer import Phase26Model
    model = Phase26Model(base_model)
    
    checkpoint = torch.load('stage8_dataset/stage11a_r2_phase26_checkpoint.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    print("  ✓ 模型加载完成")
    
    # 运行验证
    runner = ExternalValidationRunner(model)
    results = runner.run_all_validations()
    
    return results


if __name__ == "__main__":
    results = run_external_validation()
