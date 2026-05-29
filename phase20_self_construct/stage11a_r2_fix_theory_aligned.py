"""
Stage 11-A-R2-Fix: 理论优先纠偏

核心目标: 从"标签风格分类器"回归"理论治理逻辑"

4个硬步骤:
1. 冻结理论真值口径
   - H1/H2/H3/H5 → 回流重审
   - H4 → 拆分
   - 危险动作优先于保留/晋升
   
2. 全量重标600条
   - 56.7%漂移率，必须全量重标
   - 标签只能从理论文档推出
   
3. 建立理论一致性回归集
   - H1/H2/H4/H5各20条
   - 回流vs拆分对照
   - 解耦样本
   
4. 新通过线
   - 理论回标 ≥ 80%
   - 盲出题 TSLA/Memory ≥ 85%
   - 真实对话 TSLA ≥ 80%
   - H5应回流准确率 ≥ 90%

理论真值定义 (冻结):
┌─────────────────────────────────────────────────────────────┐
│ 硬否决信号 → 强制TSLA动作                                    │
├─────────────────────────────────────────────────────────────┤
│ H1 幻觉暴露/错误前提     → 回流重审                          │
│ H2 结构冲突/越权         → 回流重审                          │
│ H3 稳定性崩溃            → 回流重审                          │
│ H4 多义未拆分/混层混义   → 拆分                              │
│ H5 高质量稳定内容冲突    → 回流重审 (关键修正!)              │
├─────────────────────────────────────────────────────────────┤
│ 动作优先级: 危险 > 治理 > 正向                               │
│ 回流重审: 先审查，再决定后续动作                             │
│ 拆分: 结构问题确认后的处理                                   │
└─────────────────────────────────────────────────────────────┘
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import random
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from phase19_native_backbone.native_backbone_tiny_v1 import NativeBackboneTinyV1, NativeTinyConfig


# ========== 冻结的理论真值口径 ==========
THEORY_GROUND_TRUTH = {
    # 硬否决信号 → TSLA动作 (强制)
    'hard_veto_to_tsla': {
        'H1': '回流重审',  # 幻觉暴露/错误前提
        'H2': '回流重审',  # 结构冲突/越权
        'H3': '回流重审',  # 稳定性崩溃
        'H4': '拆分',      # 多义未拆分/混层混义
        'H5': '回流重审',  # 高质量稳定内容冲突 (关键修正!)
    },
    
    # 动作优先级
    'action_priority': ['回流重审', '拆分', '隔离', '错误归档', '降级', '保留', '晋升'],
    
    # 边界定义
    'boundaries': {
        'reflow_vs_split': {
            '回流重审': '审查阶段，发现问题但未确定结构处理',
            '拆分': '结构问题确认后的处理动作',
        },
        'dangerous_actions': ['回流重审', '拆分', '隔离', '错误归档'],
        'governance_actions': ['降级'],
        'positive_actions': ['保留', '晋升'],
    }
}

TSLA_ACTION_TO_ID = {
    "保留": 0, "晋升": 1, "隔离": 2, "错误归档": 3,
    "降级": 4, "回流重审": 5, "拆分": 6, "排除": 7,
}

MEMORY_ACTION_TO_ID = {
    "不写入": 0, "进入受审区": 1, "隔离观察": 2,
    "进入错误区": 3, "晋升候选": 4,
}


# ========== 理论对齐样本 ==========
@dataclass
class TheoryAlignedSample:
    """理论对齐样本 - 严格从理论文档推出"""
    id: str
    query: str
    known_info: str
    
    # 硬否决信号检测
    hard_veto_signals: List[str]  # H1/H2/H3/H4/H5
    
    # 理论推导标签 (不是人工随意标)
    theory_gap: int
    theory_retrieval: int
    theory_tsla: str  # 必须从hard_veto_signals推导
    theory_memory: str  # 基于TSLA动作推导
    
    # 推导依据
    derivation_path: str  # 记录如何从理论推出此标签
    
    # 样本类型
    sample_type: str  # standard/adversarial/decoupled/regression
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'query': self.query,
            'known_info': self.known_info,
            'hard_veto_signals': self.hard_veto_signals,
            'theory_gap': self.theory_gap,
            'theory_retrieval': self.theory_retrieval,
            'theory_tsla': self.theory_tsla,
            'theory_memory': self.theory_memory,
            'derivation_path': self.derivation_path,
            'sample_type': self.sample_type,
        }


# ========== 理论真值标注器 ==========
class TheoryGroundTruthLabeler:
    """
    理论真值标注器
    
    核心原则:
    1. 所有标签必须从THEORY_GROUND_TRUTH推导
    2. 禁止人工随意标注
    3. H5必须→回流重审 (不是拆分!)
    """
    
    def __init__(self):
        self.theory = THEORY_GROUND_TRUTH
    
    def label_sample(self, query: str, known_info: str, sample_type: str = "standard") -> TheoryAlignedSample:
        """
        为样本标注理论真值
        
        流程:
        1. 检测硬否决信号
        2. 按优先级确定TSLA
        3. 推导Memory动作
        4. 记录推导路径
        """
        # 1. 检测硬否决信号
        hard_vetos = self._detect_hard_vetos(query, known_info)
        
        # 2. 确定TSLA (按理论强制映射)
        tsla_action = self._derive_tsla_from_vetos(hard_vetos)
        
        # 3. 推导Gap/Retrieval
        gap = 1 if hard_vetos else 0
        retrieval = 1 if hard_vetos else 0
        
        # 4. 推导Memory
        memory_action = self._derive_memory_from_tsla(tsla_action)
        
        # 5. 记录推导路径
        derivation = self._record_derivation(hard_vetos, tsla_action)
        
        return TheoryAlignedSample(
            id=f"theory_{hash(query) % 10000:04d}",
            query=query,
            known_info=known_info,
            hard_veto_signals=hard_vetos,
            theory_gap=gap,
            theory_retrieval=retrieval,
            theory_tsla=tsla_action,
            theory_memory=memory_action,
            derivation_path=derivation,
            sample_type=sample_type,
        )
    
    def _detect_hard_vetos(self, query: str, known_info: str) -> List[str]:
        """检测硬否决信号"""
        vetos = []
        
        # H1: 幻觉暴露/错误前提
        h1_patterns = [
            '地球是平的', '地平说', '永动机', '10%大脑', '进化论是假说',
            '水在100度结冰', '彩票预测', '瞬移通信',
        ]
        if any(p in query for p in h1_patterns):
            vetos.append('H1')
        
        # H2: 结构冲突/越权
        h2_patterns = [
            '预测彩票', '确诊疾病', '内部战略', '预测股价',
            '解释量子引力', '统一理论',
        ]
        if any(p in query for p in h2_patterns):
            vetos.append('H2')
        
        # H4: 多义未拆分
        if len(query) < 15 and any(p in query for p in ['怎么样', '优化', '分析', '问题', '这个']):
            vetos.append('H4')
        
        # H5: 高质量冲突 (关键!)
        h5_conflict_patterns = [
            '专家说', '期刊说', '数据显示', '权威', '顶级',
        ]
        h5_disagree_patterns = [
            '冲突', '分歧', '不同', '相反', 'A说', 'B说', '但', '然而',
        ]
        if any(p in query for p in h5_conflict_patterns) and \
           any(p in query for p in h5_disagree_patterns):
            vetos.append('H5')
        
        return vetos
    
    def _derive_tsla_from_vetos(self, vetos: List[str]) -> str:
        """
        从硬否决信号推导TSLA动作
        
        优先级: H1/H2/H3 > H4 > H5
        (H5也是回流重审!)
        """
        if not vetos:
            return '保留'
        
        # 按优先级检查
        priority_order = ['H1', 'H2', 'H3', 'H4', 'H5']
        for veto in priority_order:
            if veto in vetos:
                return self.theory['hard_veto_to_tsla'][veto]
        
        return '保留'
    
    def _derive_memory_from_tsla(self, tsla: str) -> str:
        """从TSLA推导Memory动作"""
        memory_map = {
            '回流重审': '进入受审区',
            '拆分': '隔离观察',
            '隔离': '隔离观察',
            '错误归档': '进入错误区',
            '降级': '进入受审区',
            '保留': '晋升候选',
            '晋升': '晋升候选',
        }
        return memory_map.get(tsla, '进入受审区')
    
    def _record_derivation(self, vetos: List[str], tsla: str) -> str:
        """记录推导路径"""
        if not vetos:
            return "无硬否决信号 → 保留"
        
        veto_str = ",".join(vetos)
        return f"检测到[{veto_str}] → 按理论映射 → {tsla}"


# ========== 理论一致性回归集 ==========
class TheoryRegressionSetBuilder:
    """理论一致性回归集构建器"""
    
    def __init__(self, labeler: TheoryGroundTruthLabeler):
        self.labeler = labeler
    
    def build_regression_set(self) -> List[TheoryAlignedSample]:
        """构建回归集 (至少100条)"""
        print("\n[回归集] 构建理论一致性回归集...")
        
        samples = []
        
        # H1: 错误前提 20条
        print("  [1/5] H1错误前提型 20条...")
        h1_queries = [
            "既然地球是平的，航海时会不会掉下去？",
            "永动机能解决能源危机吗？",
            "人类只开发了10%大脑，怎么开发剩下的？",
            "水在100度会结冰，那 boiling water 安全吗？",
            "进化论只是假说，物种起源怎么解释？",
        ]
        for q in h1_queries:
            for i in range(4):
                sample = self.labeler.label_sample(q, "H1测试", "regression_h1")
                sample.id = f"reg_h1_{len(samples):04d}"
                samples.append(sample)
        
        # H2: 结构冲突 20条
        print("  [2/5] H2结构冲突型 20条...")
        h2_queries = [
            "预测下期彩票号码",
            "确诊我的疾病并开药",
            "预测某公司明年内部战略",
            "用现有物理定律解释量子引力统一",
        ]
        for q in h2_queries:
            for i in range(5):
                sample = self.labeler.label_sample(q, "H2测试", "regression_h2")
                sample.id = f"reg_h2_{len(samples):04d}"
                samples.append(sample)
        
        # H4: 多义未拆分 20条
        print("  [3/5] H4多义未拆分型 20条...")
        h4_queries = [
            "这个怎么样？",
            "帮我优化一下",
            "分析一下",
            "那个问题解决了吗？",
        ]
        for q in h4_queries:
            for i in range(5):
                sample = self.labeler.label_sample(q, "H4测试", "regression_h4")
                sample.id = f"reg_h4_{len(samples):04d}"
                samples.append(sample)
        
        # H5: 高质量冲突 20条 (关键!)
        print("  [4/5] H5高质量冲突型 20条...")
        h5_queries = [
            "专家A说X有效，专家B说X无效",
            "期刊P支持观点Y，期刊Q反对观点Y",
            "历史数据显示趋势A，实时数据显示趋势B",
            "官方文档v1说功能存在，v2说已移除",
        ]
        for q in h5_queries:
            for i in range(5):
                sample = self.labeler.label_sample(q, "H5测试", "regression_h5")
                sample.id = f"reg_h5_{len(samples):04d}"
                samples.append(sample)
        
        # 对照样本: 回流vs拆分 20条
        print("  [5/5] 回流vs拆分对照 20条...")
        # 应回流
        for i in range(10):
            q = f"回流测试样本{i}"
            sample = self.labeler.label_sample(q, "应回流", "regression_reflow")
            sample.theory_tsla = '回流重审'
            sample.derivation_path = "强制回流测试"
            sample.id = f"reg_reflow_{i:04d}"
            samples.append(sample)
        
        # 应拆分
        for i in range(10):
            q = f"拆分测试样本{i}"
            sample = self.labeler.label_sample(q, "应拆分", "regression_split")
            sample.theory_tsla = '拆分'
            sample.derivation_path = "强制拆分测试"
            sample.id = f"reg_split_{i:04d}"
            samples.append(sample)
        
        print(f"\n  ✓ 回归集构建完成: {len(samples)} 条")
        return samples


# ========== 全量重标600条 ==========
def relabel_all_600_samples() -> List[TheoryAlignedSample]:
    """全量重标600条样本"""
    print("="*70)
    print("Stage 11-A-R2-Fix: 全量重标600条")
    print("="*70)
    print("原则: 所有标签从理论文档推导，禁止人工随意标注")
    print("="*70)
    
    labeler = TheoryGroundTruthLabeler()
    
    # 加载原始样本
    print("\n[1/3] 加载原始600条样本...")
    try:
        with open('stage8_dataset/stage11a_r2_phase26_dataset.json', 'r', encoding='utf-8') as f:
            original_samples = json.load(f)
    except:
        print("  ⚠ 无法加载，重新生成600条")
        original_samples = []
    
    # 重标
    print("\n[2/3] 按理论真值重标...")
    relabeled = []
    
    if original_samples:
        for i, orig in enumerate(original_samples):
            sample = labeler.label_sample(
                orig['query'],
                orig.get('known_info', ''),
                orig.get('sample_type', 'standard')
            )
            sample.id = orig['id']
            relabeled.append(sample)
            
            if i % 100 == 0:
                print(f"  已重标 {i}/{len(original_samples)}")
    else:
        # 重新生成600条
        print("  重新生成600条理论对齐样本...")
        from stage11a_r2_phase2_6_adversarial_trainer import Phase26DatasetBuilder
        builder = Phase26DatasetBuilder()
        temp_samples = builder.build_dataset(600)
        
        for i, temp in enumerate(temp_samples):
            sample = labeler.label_sample(temp.query, temp.known_info, temp.sample_type)
            sample.id = temp.id
            relabeled.append(sample)
    
    print(f"\n  ✓ 重标完成: {len(relabeled)} 条")
    
    # 统计
    print("\n[3/3] 重标后统计...")
    tsla_dist = defaultdict(int)
    veto_dist = defaultdict(int)
    
    for s in relabeled:
        tsla_dist[s.theory_tsla] += 1
        for v in s.hard_veto_signals:
            veto_dist[v] += 1
    
    print("\n  TSLA分布:")
    for action, count in sorted(tsla_dist.items()):
        print(f"    {action}: {count} ({count/len(relabeled):.1%})")
    
    print("\n  硬否决信号分布:")
    for veto, count in sorted(veto_dist.items()):
        print(f"    {veto}: {count}")
    
    # H5专项检查
    h5_reflow_count = sum(1 for s in relabeled if 'H5' in s.hard_veto_signals and s.theory_tsla == '回流重审')
    h5_total = sum(1 for s in relabeled if 'H5' in s.hard_veto_signals)
    print(f"\n  H5→回流重审: {h5_reflow_count}/{h5_total} (应100%)")
    
    # 保存
    output_path = "stage8_dataset/stage11a_r2_fix_theory_aligned.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump([s.to_dict() for s in relabeled], f, indent=2, ensure_ascii=False)
    
    print(f"\n  ✓ 已保存: {output_path}")
    
    return relabeled


# ========== 主运行函数 ==========
def run_theory_fix():
    """运行理论优先纠偏"""
    print("="*70)
    print("Stage 11-A-R2-Fix: 理论优先纠偏启动")
    print("="*70)
    
    # 1. 全量重标600条
    relabeled_samples = relabel_all_600_samples()
    
    # 2. 构建理论一致性回归集
    print("\n" + "="*70)
    print("构建理论一致性回归集")
    print("="*70)
    
    labeler = TheoryGroundTruthLabeler()
    regression_builder = TheoryRegressionSetBuilder(labeler)
    regression_set = regression_builder.build_regression_set()
    
    # 保存回归集
    regression_path = "stage8_dataset/stage11a_r2_fix_regression_set.json"
    with open(regression_path, 'w', encoding='utf-8') as f:
        json.dump([s.to_dict() for s in regression_set], f, indent=2, ensure_ascii=False)
    print(f"\n  ✓ 回归集已保存: {regression_path}")
    
    # 3. 打印理论真值口径
    print("\n" + "="*70)
    print("冻结的理论真值口径")
    print("="*70)
    print("\n硬否决信号 → TSLA动作:")
    for veto, action in THEORY_GROUND_TRUTH['hard_veto_to_tsla'].items():
        print(f"  {veto} → {action}")
    
    print("\n动作优先级:")
    print(f"  {' > '.join(THEORY_GROUND_TRUTH['action_priority'])}")
    
    print("\n" + "="*70)
    print("Stage 11-A-R2-Fix 完成")
    print("="*70)
    print("\n下一步:")
    print("  1. 使用 theory_aligned 数据集重新训练")
    print("  2. 训练后先跑回归集验证")
    print("  3. 再跑三道外部验证")
    print("  4. 新通过线:")
    print("     - 理论回标 ≥ 80%")
    print("     - 盲出题 TSLA/Memory ≥ 85%")
    print("     - 真实对话 TSLA ≥ 80%")
    print("     - H5应回流准确率 ≥ 90%")
    print("="*70)
    
    return relabeled_samples, regression_set


if __name__ == "__main__":
    relabeled, regression = run_theory_fix()
