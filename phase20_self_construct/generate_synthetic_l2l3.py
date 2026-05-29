"""
生成合成 L2/L3 数据集
用于 Stage 8 测试 (当真实数据集无法下载时)
"""

import json
import random
from pathlib import Path
from typing import List, Dict


# L2 级别问题模板 (需要简单推理)
L2_TEMPLATES = [
    {
        "question": "如果 {obj} 放在 {place} 里，它会怎么样？",
        "context": "{obj} 是一种 {property} 的物品。{place} 的环境是 {env}。",
        "answer": "它会 {result}",
        "reasoning": ["识别物品属性", "分析环境影响", "得出结论"],
    },
    {
        "question": "为什么 {animal} 会 {action}？",
        "context": "{animal} 生活在 {habitat}。它们需要 {need} 来生存。",
        "answer": "因为 {reason}",
        "reasoning": ["理解动物习性", "分析生存需求", "解释行为原因"],
    },
    {
        "question": "{person} 应该怎么做才能 {goal}？",
        "context": "{person} 目前面临 {situation}。{resource} 可以帮助实现目标。",
        "answer": "应该 {solution}",
        "reasoning": ["分析当前状况", "评估可用资源", "制定解决方案"],
    },
]

# L3 级别问题模板 (需要复杂推理)
L3_TEMPLATES = [
    {
        "question": "考虑到 {factor1} 和 {factor2}，{subject} 的最佳策略是什么？",
        "context": "{subject} 面临 {challenge}。{factor1} 意味着 {implication1}。{factor2} 意味着 {implication2}。",
        "answer": "最佳策略是 {strategy}",
        "reasoning": ["分析多因素影响", "权衡利弊", "综合判断", "制定策略"],
    },
    {
        "question": "如果 {condition} 发生，对 {system} 会产生什么连锁反应？",
        "context": "{system} 包含 {component1} 和 {component2}。{condition} 会影响 {component1}，进而影响 {component2}。",
        "answer": "会导致 {consequence}",
        "reasoning": ["识别系统组件", "分析初始影响", "追踪连锁反应", "预测最终结果"],
    },
    {
        "question": "比较 {option1} 和 {option2}，在 {criteria} 方面哪个更优？",
        "context": "{option1} 的特点是 {feature1}。{option2} 的特点是 {feature2}。评判标准是 {criteria}。",
        "answer": "{better_option} 更优，因为 {justification}",
        "reasoning": ["明确评判标准", "分析选项特征", "对比优劣", "得出结论"],
    },
]

# 填充词库
FILLERS = {
    "obj": ["冰块", "金属", "纸张", "水果", "电池"],
    "place": ["冰箱", "烤箱", "水中", "阳光下", "密封容器"],
    "property": ["易融化", "导热快", "易燃", "易腐烂", "会放电"],
    "env": ["低温", "高温", "潮湿", "干燥", "密闭"],
    "result": ["保持固态", "融化", "生锈", "变干", "放电"],
    "animal": ["企鹅", "骆驼", "蝙蝠", "海豚", "松鼠"],
    "action": ["迁徙", "储存食物", "冬眠", "群居", "筑巢"],
    "habitat": ["极地", "沙漠", "洞穴", "海洋", "森林"],
    "need": ["保暖", "储水", "避光", "氧气", "食物储备"],
    "reason": ["适应环境", "生存本能", "季节变化", "繁殖需要", "躲避天敌"],
    "person": ["学生", "医生", "工程师", "农民", "商人"],
    "goal": ["提高效率", "节省成本", "改善健康", "增加产量", "扩大市场"],
    "situation": ["时间紧迫", "资源有限", "竞争激烈", "技术落后", "需求变化"],
    "resource": ["新技术", "团队合作", "数据分析", "专业知识", "资金支持"],
    "solution": ["优化流程", "学习新技能", "寻求帮助", "改变策略", "投资升级"],
    "factor1": ["成本", "时间", "质量", "风险", "市场需求"],
    "factor2": ["技术可行性", "团队能力", "竞争态势", "政策环境", "用户反馈"],
    "subject": ["公司", "项目", "产品", "团队", "个人发展"],
    "challenge": ["市场萎缩", "技术瓶颈", "人才流失", "资金短缺", "品牌危机"],
    "implication1": ["需要控制预算", "必须加快进度", "不能妥协质量", "需要规避风险", "要紧跟趋势"],
    "implication2": ["技术是关键", "人才是核心", "竞争很激烈", "政策要遵守", "用户是上帝"],
    "strategy": ["分阶段实施", "聚焦核心优势", "寻求合作", "创新突破", "稳健发展"],
    "condition": ["政策变化", "技术革新", "市场崩溃", "自然灾害", "竞争对手行动"],
    "system": ["供应链", "生态系统", "经济体系", "社会网络", "技术架构"],
    "component1": ["生产环节", "初级消费者", "金融机构", "信息节点", "前端模块"],
    "component2": ["分销渠道", "次级消费者", "实体经济", "传播路径", "后端服务"],
    "consequence": ["系统性风险", "连锁倒闭", "经济衰退", "信息中断", "服务瘫痪"],
    "option1": ["方案A", "传统方法", "保守策略", "短期计划", "单一方案"],
    "option2": ["方案B", "创新方法", "激进策略", "长期规划", "综合方案"],
    "criteria": ["成本效益", "可持续性", "风险控制", "用户体验", "长期价值"],
    "feature1": ["成本低", "成熟稳定", "风险小", "见效快", "简单易行"],
    "feature2": ["效率高", "创新性强", "潜力大", "基础扎实", "全面系统"],
    "better_option": ["方案B", "创新方法", "激进策略", "长期规划", "综合方案"],
    "justification": ["更符合长期利益", "适应性更强", "风险调整后收益更高", "用户满意度更高", "综合价值更大"],
}


def generate_l2_sample(sample_id: int) -> Dict:
    """生成 L2 级别样本"""
    template = random.choice(L2_TEMPLATES)
    
    # 填充模板
    filled = {}
    for key in ["question", "context", "answer"]:
        text = template[key]
        for filler_key in FILLERS:
            if f"{{{filler_key}}}" in text:
                text = text.replace(f"{{{filler_key}}}", random.choice(FILLERS[filler_key]))
        filled[key] = text
    
    return {
        "sample_id": f"syn_l2_{sample_id:05d}",
        "question": filled["question"],
        "context": filled["context"],
        "answer": filled["answer"],
        "difficulty": "L2",
        "expected_gap": 1,
        "expected_retrieval": 1,
        "expected_policy": 1,
        "knowledge_units": [
            {"type": "fact", "content": filled["context"][:50], "relation": "background"},
        ],
        "reasoning_chain": template["reasoning"],
        "source": "synthetic_l2",
    }


def generate_l3_sample(sample_id: int) -> Dict:
    """生成 L3 级别样本"""
    template = random.choice(L3_TEMPLATES)
    
    # 填充模板
    filled = {}
    for key in ["question", "context", "answer"]:
        text = template[key]
        for filler_key in FILLERS:
            if f"{{{filler_key}}}" in text:
                text = text.replace(f"{{{filler_key}}}", random.choice(FILLERS[filler_key]))
        filled[key] = text
    
    return {
        "sample_id": f"syn_l3_{sample_id:05d}",
        "question": filled["question"],
        "context": filled["context"],
        "answer": filled["answer"],
        "difficulty": "L3",
        "expected_gap": 1,
        "expected_retrieval": 1,
        "expected_policy": 1,
        "knowledge_units": [
            {"type": "fact", "content": filled["context"][:50], "relation": "background"},
            {"type": "rule", "content": "需要多步推理", "relation": "reasoning"},
        ],
        "reasoning_chain": template["reasoning"],
        "source": "synthetic_l3",
    }


def generate_synthetic_dataset(
    num_l2: int = 500,
    num_l3: int = 300,
    output_dir: str = "stage8_dataset",
) -> Dict:
    """生成合成数据集"""
    print("="*70)
    print("生成合成 L2/L3 数据集")
    print("="*70)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 生成 L2 样本
    print(f"\n[1/2] 生成 {num_l2} 条 L2 样本...")
    l2_samples = [generate_l2_sample(i) for i in range(num_l2)]
    
    l2_file = output_path / "synthetic_l2.jsonl"
    with open(l2_file, 'w', encoding='utf-8') as f:
        for sample in l2_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"  ✓ 保存到: {l2_file}")
    
    # 生成 L3 样本
    print(f"\n[2/2] 生成 {num_l3} 条 L3 样本...")
    l3_samples = [generate_l3_sample(i) for i in range(num_l3)]
    
    l3_file = output_path / "synthetic_l3.jsonl"
    with open(l3_file, 'w', encoding='utf-8') as f:
        for sample in l3_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"  ✓ 保存到: {l3_file}")
    
    # 合并文件
    combined_file = output_path / "synthetic_all.jsonl"
    with open(combined_file, 'w', encoding='utf-8') as f:
        for sample in l2_samples + l3_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"  ✓ 合并保存到: {combined_file}")
    
    print("\n" + "="*70)
    print("生成完成！")
    print("="*70)
    print(f"L2 样本: {num_l2} 条")
    print(f"L3 样本: {num_l3} 条")
    print(f"总计: {num_l2 + num_l3} 条")
    print("="*70)
    
    return {
        "l2_file": str(l2_file),
        "l3_file": str(l3_file),
        "combined_file": str(combined_file),
        "num_l2": num_l2,
        "num_l3": num_l3,
    }


if __name__ == "__main__":
    generate_synthetic_dataset()
