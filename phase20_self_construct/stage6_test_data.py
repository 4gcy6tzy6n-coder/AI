"""
Stage 6 Test Data Generator

Phase 3 端到端测试数据集

包含:
1. 单轮查询测试数据
2. 多轮对话测试数据
3. 复杂场景测试数据
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class TestQuery:
    """测试查询"""
    query_id: str
    query_text: str
    expected_gap: int  # 期望检测到的 GAP
    category: str  # 类别
    description: str = ""


@dataclass
class MultiTurnConversation:
    """多轮对话"""
    conversation_id: str
    turns: List[TestQuery]
    expected_outcome: str
    description: str = ""


# ==================== 单轮查询测试数据 ====================

SINGLE_TURN_QUERIES: List[TestQuery] = [
    # 基础查询
    TestQuery("basic_001", "你好", 0, "basic", "基础问候"),
    TestQuery("basic_002", "今天天气怎么样", 0, "basic", "日常查询"),
    TestQuery("basic_003", "帮我查个资料", 0, "basic", "简单请求"),
    
    # GAP 类型 1: 知识缺口
    TestQuery("gap1_001", "请解释量子计算的最新进展", 1, "gap1", "前沿知识查询"),
    TestQuery("gap1_002", "2024年AI领域有什么重大突破", 1, "gap1", "时效性知识"),
    TestQuery("gap1_003", "最新的深度学习架构是什么", 1, "gap1", "技术知识"),
    TestQuery("gap1_004", "如何优化大模型的推理速度", 1, "gap1", "技术问题"),
    TestQuery("gap1_005", "Transformer架构的变体有哪些", 1, "gap1", "架构知识"),
    
    # GAP 类型 2: 策略缺口
    TestQuery("gap2_001", "我需要在5分钟内完成这个任务", 2, "gap2", "时间约束"),
    TestQuery("gap2_002", "如何在资源受限的情况下优化", 2, "gap2", "资源约束"),
    TestQuery("gap2_003", "请给出最高效的解决方案", 2, "gap2", "效率优化"),
    TestQuery("gap2_004", "如何在多目标间平衡", 2, "gap2", "多目标优化"),
    TestQuery("gap2_005", "紧急情况下的处理策略", 2, "gap2", "紧急策略"),
    
    # GAP 类型 3: 治理缺口
    TestQuery("gap3_001", "这个请求可能涉及隐私问题", 3, "gap3", "隐私治理"),
    TestQuery("gap3_002", "如何处理敏感信息", 3, "gap3", "敏感信息"),
    TestQuery("gap3_003", "这个决策的伦理考量是什么", 3, "gap3", "伦理治理"),
    TestQuery("gap3_004", "如何确保公平性", 3, "gap3", "公平性治理"),
    TestQuery("gap3_005", "数据使用的合规性检查", 3, "gap3", "合规治理"),
    
    # GAP 类型 4: 反馈缺口
    TestQuery("gap4_001", "请总结我们之前的对话", 4, "gap4", "对话总结"),
    TestQuery("gap4_002", "根据之前的讨论给出建议", 4, "gap4", "上下文建议"),
    TestQuery("gap4_003", "基于历史记录优化方案", 4, "gap4", "历史优化"),
    TestQuery("gap4_004", "回顾之前的决策", 4, "gap4", "决策回顾"),
    TestQuery("gap4_005", "从过去经验中学习", 4, "gap4", "经验学习"),
]


# ==================== 多轮对话测试数据 ====================

MULTI_TURN_CONVERSATIONS: List[MultiTurnConversation] = [
    # 对话 1: 知识探索
    MultiTurnConversation(
        "conv_knowledge_001",
        [
            TestQuery("conv1_t1", "什么是机器学习", 1, "knowledge"),
            TestQuery("conv1_t2", "深度学习有什么特点", 1, "knowledge"),
            TestQuery("conv1_t3", "Transformer为什么有效", 1, "knowledge"),
            TestQuery("conv1_t4", "如何训练大语言模型", 1, "knowledge"),
        ],
        "知识累积",
        "知识探索对话"
    ),
    
    # 对话 2: 问题解决
    MultiTurnConversation(
        "conv_problem_001",
        [
            TestQuery("conv2_t1", "我的模型过拟合了", 1, "problem"),
            TestQuery("conv2_t2", "有什么正则化方法", 1, "problem"),
            TestQuery("conv2_t3", "Dropout和BatchNorm哪个更好", 2, "problem"),
            TestQuery("conv2_t4", "如何在验证集上评估", 4, "problem"),
        ],
        "问题解决",
        "问题诊断对话"
    ),
    
    # 对话 3: 策略优化
    MultiTurnConversation(
        "conv_strategy_001",
        [
            TestQuery("conv3_t1", "如何优化推理速度", 2, "strategy"),
            TestQuery("conv3_t2", "量化会有什么影响", 2, "strategy"),
            TestQuery("conv3_t3", "模型压缩的方法", 2, "strategy"),
            TestQuery("conv3_t4", "精度和速度的平衡", 2, "strategy"),
        ],
        "策略优化",
        "性能优化对话"
    ),
    
    # 对话 4: 治理讨论
    MultiTurnConversation(
        "conv_governance_001",
        [
            TestQuery("conv4_t1", "AI系统的安全性如何保证", 3, "governance"),
            TestQuery("conv4_t2", "隐私保护的技术手段", 3, "governance"),
            TestQuery("conv4_t3", "如何防止偏见", 3, "governance"),
            TestQuery("conv4_t4", "可解释性的重要性", 3, "governance"),
        ],
        "治理讨论",
        "AI治理对话"
    ),
    
    # 对话 5: 混合场景
    MultiTurnConversation(
        "conv_mixed_001",
        [
            TestQuery("conv5_t1", "解释神经网络", 1, "mixed"),
            TestQuery("conv5_t2", "如何高效训练", 2, "mixed"),
            TestQuery("conv5_t3", "训练数据的安全", 3, "mixed"),
            TestQuery("conv5_t4", "根据之前讨论总结", 4, "mixed"),
        ],
        "混合场景",
        "综合对话"
    ),
]


# ==================== 复杂场景测试数据 ====================

COMPLEX_SCENARIOS: List[Dict] = [
    {
        "id": "complex_multi_gap",
        "name": "多 GAP 并发",
        "description": "单个查询触发多个 GAP",
        "queries": [
            "如何在保护隐私的前提下，用最高效的方法学习最新的AI技术",
            "请基于历史对话，给出兼顾安全性和性能的资源受限优化方案",
        ],
        "expected_gaps": [1, 2, 3],  # 可能同时触发多个 GAP
    },
    {
        "id": "complex_rollback",
        "name": "回滚场景",
        "description": "需要触发回滚的场景",
        "queries": [
            "执行一个可能导致系统不稳定的操作",
            "请尝试一个高风险的新策略",
        ],
        "expected_behavior": "rollback_triggered",
    },
    {
        "id": "complex_stress",
        "name": "压力场景",
        "description": "高频率查询测试",
        "queries": [f"压力测试查询 {i}" for i in range(20)],
        "expected_behavior": "stable_under_load",
    },
    {
        "id": "complex_edge",
        "name": "边界场景",
        "description": "边界情况测试",
        "queries": [
            "",  # 空查询
            "a",  # 极短查询
            "x" * 1000,  # 超长查询
            "!@#$%^&*()",  # 特殊字符
        ],
        "expected_behavior": "graceful_handling",
    },
]


# ==================== 数据生成函数 ====================

def get_single_turn_queries(category: str = None) -> List[TestQuery]:
    """获取单轮查询"""
    if category:
        return [q for q in SINGLE_TURN_QUERIES if q.category == category]
    return SINGLE_TURN_QUERIES


def get_multi_turn_conversations() -> List[MultiTurnConversation]:
    """获取多轮对话"""
    return MULTI_TURN_CONVERSATIONS


def get_complex_scenarios() -> List[Dict]:
    """获取复杂场景"""
    return COMPLEX_SCENARIOS


def generate_test_suite() -> Dict:
    """生成完整测试套件"""
    return {
        'single_turn': {
            'total': len(SINGLE_TURN_QUERIES),
            'by_category': {
                'basic': len([q for q in SINGLE_TURN_QUERIES if q.category == 'basic']),
                'gap1': len([q for q in SINGLE_TURN_QUERIES if q.category == 'gap1']),
                'gap2': len([q for q in SINGLE_TURN_QUERIES if q.category == 'gap2']),
                'gap3': len([q for q in SINGLE_TURN_QUERIES if q.category == 'gap3']),
                'gap4': len([q for q in SINGLE_TURN_QUERIES if q.category == 'gap4']),
            },
            'queries': SINGLE_TURN_QUERIES,
        },
        'multi_turn': {
            'total': len(MULTI_TURN_CONVERSATIONS),
            'conversations': MULTI_TURN_CONVERSATIONS,
        },
        'complex': {
            'total': len(COMPLEX_SCENARIOS),
            'scenarios': COMPLEX_SCENARIOS,
        },
    }


# ==================== 统计信息 ====================

def print_test_suite_summary():
    """打印测试套件摘要"""
    suite = generate_test_suite()
    
    print("=" * 70)
    print("Stage 6 Phase 3 测试数据集摘要")
    print("=" * 70)
    
    print("\n【单轮查询】")
    print(f"  总计: {suite['single_turn']['total']}")
    for cat, count in suite['single_turn']['by_category'].items():
        print(f"    {cat}: {count}")
    
    print("\n【多轮对话】")
    print(f"  总计: {suite['multi_turn']['total']}")
    for conv in suite['multi_turn']['conversations']:
        print(f"    {conv.conversation_id}: {len(conv.turns)} 轮")
    
    print("\n【复杂场景】")
    print(f"  总计: {suite['complex']['total']}")
    for scenario in suite['complex']['scenarios']:
        print(f"    {scenario['id']}: {scenario['name']}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    print_test_suite_summary()
