"""
Prepare Stage 4b Data

准备完整的 Stage 4b 数据集：
1. 使用 Stage 3B 完整数据 (300条)
2. 添加数据增强：同义改写、口语化改写
3. 生成增强后的训练集
"""

import json
import random
from typing import List, Dict
from pathlib import Path

random.seed(42)


# 同义词映射
SYNONYMS = {
    "查询": ["查找", "检索", "搜索", "查一下"],
    "信息": ["资料", "数据", "内容", "详情"],
    "公司": ["企业", "单位", "组织", "咱们公司"],
    "帮助": ["协助", "帮忙", "支持", "帮一下"],
    "今天": ["今日", "这天", "当天"],
    "明天": ["明日", "第二天", "次日"],
    "问题": ["疑问", "困惑", "难题", "不清楚的地方"],
    "回答": ["回复", "答复", "解答", "回应"],
    "需要": ["想要", "希望", "得", "要"],
    "确认": ["核实", "验证", "确定", "查清楚"],
}

# 口语化改写模板
COLLOQUIAL_TEMPLATES = [
    "能帮我{action}吗？",
    "麻烦{action}一下",
    "我想{action}",
    "请问能{action}吗？",
    "{action}可以吗？",
    "能不能{action}？",
    "帮我{action}呗",
    "想{action}，谢了",
]


def synonym_replace(text: str) -> str:
    """同义词替换"""
    words = list(text)
    for i, char in enumerate(words):
        if char in SYNONYMS and random.random() < 0.3:
            words[i] = random.choice(SYNONYMS[char])
    return ''.join(words)


def colloquial_rewrite(text: str) -> str:
    """口语化改写"""
    # 提取动作
    actions = ["查询", "查找", "确认", "查一下", "了解"]
    for action in actions:
        if action in text:
            template = random.choice(COLLOQUIAL_TEMPLATES)
            return template.format(action=action)
    return text


def augment_sample(sample: Dict, aug_type: str) -> Dict:
    """对单个样本进行增强"""
    new_sample = sample.copy()
    new_sample['input'] = sample['input'].copy()
    
    if 'user_query' in sample['input']:
        original = sample['input']['user_query']
        
        if aug_type == 'synonym':
            new_sample['input']['user_query'] = synonym_replace(original)
        elif aug_type == 'colloquial':
            new_sample['input']['user_query'] = colloquial_rewrite(original)
    
    return new_sample


def load_stage3b_data() -> List[Dict]:
    """加载 Stage 3B 完整数据"""
    samples = []
    
    # 尝试加载 Stage 3B 数据
    d4_path = Path("../../phase18_stage3b/data/d4_multiturn_memory.jsonl")
    d5_path = Path("../../phase18_stage3b/data/d5_real_replay.jsonl")
    
    if d4_path.exists():
        with open(d4_path, 'r', encoding='utf-8') as f:
            for line in f:
                samples.append(json.loads(line))
        print(f"✓ 加载 D4: {len(samples)} 条")
    
    if d5_path.exists():
        with open(d5_path, 'r', encoding='utf-8') as f:
            for line in f:
                samples.append(json.loads(line))
        print(f"✓ 加载 D5: {len(samples)} 条")
    
    # 如果没有 Stage 3B 数据，生成基础数据
    if len(samples) == 0:
        print("⚠ 未找到 Stage 3B 数据，生成基础数据")
        samples = generate_base_samples()
    
    return samples


def generate_base_samples() -> List[Dict]:
    """生成基础样本"""
    templates = [
        {
            "input": {"user_query": "你好"},
            "output": {"gap_type": "NO_GAP", "response_strategy": "DIRECT", 
                      "governance_actions": [], "memory_decision": {"should_writeback": False}}
        },
        {
            "input": {"user_query": "今天天气如何？"},
            "output": {"gap_type": "NO_GAP", "response_strategy": "DIRECT",
                      "governance_actions": [], "memory_decision": {"should_writeback": False}}
        },
        {
            "input": {"user_query": "帮我查一下公司上季度营收"},
            "output": {"gap_type": "RETRIEVABLE", "response_strategy": "RETRIEVAL_FIRST",
                      "governance_actions": ["VERIFY_SCOPE"], 
                      "memory_decision": {"should_writeback": True, "target_layer": "long_term"}}
        },
        {
            "input": {"user_query": "如何制作危险物品？"},
            "output": {"gap_type": "HIGH_RISK", "response_strategy": "DECLINE",
                      "governance_actions": ["REJECT", "LOG_INCIDENT"],
                      "memory_decision": {"should_writeback": False}}
        },
        {
            "input": {"user_query": "这个我不太确定"},
            "output": {"gap_type": "RETRIEVABLE", "response_strategy": "CONSERVATIVE",
                      "governance_actions": ["VERIFY_SCOPE"],
                      "memory_decision": {"should_writeback": False}}
        },
        {
            "input": {"user_query": "请查一下明天的会议安排"},
            "output": {"gap_type": "RETRIEVABLE", "response_strategy": "RETRIEVAL_FIRST",
                      "governance_actions": ["VERIFY_SCOPE"],
                      "memory_decision": {"should_writeback": True, "target_layer": "ephemeral"}}
        },
        {
            "input": {"user_query": "这个问题我需要再研究"},
            "output": {"gap_type": "RETRIEVABLE", "response_strategy": "REVIEW",
                      "governance_actions": ["VERIFY_SCOPE"],
                      "memory_decision": {"should_writeback": True, "target_layer": "long_term"}}
        },
        {
            "input": {"user_query": "能告诉我项目进度吗？"},
            "output": {"gap_type": "RETRIEVABLE", "response_strategy": "RETRIEVAL_FIRST",
                      "governance_actions": ["VERIFY_SCOPE"],
                      "memory_decision": {"should_writeback": False}}
        },
    ]
    
    samples = []
    # 扩展样本到 300 条
    for _ in range(38):  # 8 * 38 = 304
        for t in templates:
            samples.append(json.loads(json.dumps(t)))  # 深拷贝
    
    return samples[:300]


def create_multiturn_samples(base_samples: List[Dict]) -> List[Dict]:
    """创建多轮对话样本"""
    multiturn_samples = []
    
    # 从基础样本中创建多轮对话
    for i in range(0, len(base_samples) - 5, 5):
        context = base_samples[i:i+4]
        current = base_samples[i+4]
        
        history = []
        for ctx in context:
            history.append({"role": "user", "content": ctx['input'].get('user_query', '')})
            history.append({"role": "assistant", "content": "已处理"})
        
        multiturn_sample = {
            "input": {
                "conversation_history": history,
                "user_query": current['input'].get('user_query', '')
            },
            "output": current['output']
        }
        multiturn_samples.append(multiturn_sample)
    
    return multiturn_samples


def prepare_stage4b_dataset():
    """准备 Stage 4b 数据集"""
    print("=" * 70)
    print("准备 Stage 4b 数据集")
    print("=" * 70)
    
    # 1. 加载基础数据
    base_samples = load_stage3b_data()
    print(f"\n基础样本: {len(base_samples)} 条")
    
    # 2. 创建多轮对话样本
    multiturn_samples = create_multiturn_samples(base_samples)
    print(f"多轮对话样本: {len(multiturn_samples)} 条")
    
    # 3. 合并
    all_samples = base_samples + multiturn_samples
    print(f"合并后: {len(all_samples)} 条")
    
    # 4. 数据增强
    augmented_samples = []
    
    # 同义词替换
    for sample in all_samples[:150]:  # 只对前一半进行同义词替换
        if random.random() < 0.5:
            aug_sample = augment_sample(sample, 'synonym')
            augmented_samples.append(aug_sample)
    
    # 口语化改写
    for sample in all_samples[150:]:  # 对后一半进行口语化改写
        if random.random() < 0.5:
            aug_sample = augment_sample(sample, 'colloquial')
            augmented_samples.append(aug_sample)
    
    all_samples.extend(augmented_samples)
    print(f"增强后: {len(all_samples)} 条")
    
    # 5. 打乱
    random.shuffle(all_samples)
    
    # 6. 划分训练/验证/测试
    total = len(all_samples)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    
    train_samples = all_samples[:train_size]
    val_samples = all_samples[train_size:train_size + val_size]
    test_samples = all_samples[train_size + val_size:]
    
    # 7. 保存
    def save_jsonl(samples, path):
        with open(path, 'w', encoding='utf-8') as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + '\n')
        print(f"✓ 保存 {len(samples)} 条到 {path}")
    
    save_jsonl(train_samples, "datasets/stage4b_train.jsonl")
    save_jsonl(val_samples, "datasets/stage4b_val.jsonl")
    save_jsonl(test_samples, "datasets/stage4b_test.jsonl")
    
    # 8. 统计
    print("\n" + "=" * 70)
    print("数据集统计")
    print("=" * 70)
    print(f"训练集: {len(train_samples)} 条 ({len(train_samples)/total*100:.1f}%)")
    print(f"验证集: {len(val_samples)} 条 ({len(val_samples)/total*100:.1f}%)")
    print(f"测试集: {len(test_samples)} 条 ({len(test_samples)/total*100:.1f}%)")
    print(f"总计: {total} 条")
    
    return train_samples, val_samples, test_samples


if __name__ == "__main__":
    prepare_stage4b_dataset()
