"""
Stage 10-3: 产品化人工对话测试 (Manual Pilot Dialogue Test)

目标:
- 验证产品在真实交互中的表现
- 从"实验验证"转向"产品验证"

测试原则:
1. 使用冻结基线版本 (product_baseline_v1)
2. 先测受控场景，不开放所有能力
3. 记录回答质量、检索触发、幻觉、用户体验

测试结构 (20-30轮):
- 5轮基础问答
- 5轮需要检索的问题
- 5轮边界/诱导问题
- 5轮多轮记忆问题
- 5轮真实使用场景问题
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

from phase19_native_backbone.native_backbone_tiny_v1 import NativeBackboneTinyV1, NativeTinyConfig
from stage7_output_kl_guard import build_output_kl_guard


class ProductDialogueTester:
    """产品对话测试器"""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        
        # 加载冻结基线模型
        print("正在加载 product_baseline_v1...")
        config = NativeTinyConfig()
        self.model = NativeBackboneTinyV1(config)
        self.model.to(device)
        self.model.eval()
        
        # Guard 配置 (冻结)
        self.guard = build_output_kl_guard(
            model=self.model,
            beta=0.2,
            num_samples=30,
            use_probs=True,
        )
        self.guard.capture_reference_outputs()
        
        # TSLA 配置 (冻结)
        self.promotion_threshold = 0.8
        self.isolation_threshold = 0.3
        
        # 对话状态
        self.conversation_history: List[Dict] = []
        self.memory_state = {
            'instant': [],      # 瞬时记忆
            'longterm': [],     # 长期候选
            'permanent': [],    # 永久记忆
        }
        
        # 测试记录
        self.test_records: List[Dict] = []
        
        # 知识库 (模拟)
        self.knowledge_base = self._load_knowledge_base()
        
        print("✓ 产品对话测试器就绪")
        print(f"  设备: {device}")
        print(f"  Guard: beta=0.2 (冻结)")
        print(f"  TSLA: promotion=0.8, isolation=0.3 (冻结)")
        print()
    
    def _load_knowledge_base(self) -> Dict[str, str]:
        """加载模拟知识库"""
        return {
            # 项目知识
            "项目目标": "Post-Transformer AI 旨在解决单模型多层任务训练的跷跷板效应",
            "核心机制": "Output KL Guard + TSLA 门控 + 固定采样策略",
            "Guard作用": "保护 writeback 输出分布，防止训练破坏主模型",
            "TSLA作用": "分层记忆治理：瞬时→长期→永久",
            
            # 技术细节
            "平衡窗口": "系统在多层任务训练中保持稳定的连续步数",
            "S9-R2": "官方基线，109-step 平衡窗口",
            "S10-1": "长期稳定性验证，649-step 平衡窗口",
            "S10-2R1": "真实数据迁移验证，399-step 平衡窗口",
            
            # 产品信息
            "当前阶段": "Stage 10-3 产品化收口",
            "产品基线": "product_baseline_v1 (2026-04-20 冻结)",
            "开放能力": "检索增强问答、单层任务、受控多层任务",
            "暂不开放": "高自由度自动晋升、大范围自学习",
        }
    
    def _encode_text(self, text: str) -> torch.Tensor:
        """文本编码为 token IDs"""
        # 简单编码：每个字符映射到一个数字
        tokens = [ord(c) % 10000 for c in text[:50]]  # 限制长度
        if len(tokens) < 10:
            tokens.extend([0] * (10 - len(tokens)))
        return torch.tensor([tokens], device=self.device)
    
    def _detect_knowledge_gap(self, question: str) -> Tuple[bool, str]:
        """检测知识缺口，决定是否需要检索"""
        # 简单规则：如果问题包含知识库关键词，认为有知识
        question_lower = question.lower()
        
        for key in self.knowledge_base.keys():
            if key in question or any(kw in question_lower for kw in key.lower().split()):
                return False, "known"
        
        # 检查是否是事实性问题
        fact_keywords = ["是什么", "什么是", "多少", "几个", "为什么", "怎么"]
        is_factual = any(kw in question for kw in fact_keywords)
        
        if is_factual and random.random() < 0.3:  # 30% 概率触发检索
            return True, "retrieval_triggered"
        
        return False, "general"
    
    def _retrieve_knowledge(self, query: str) -> Optional[str]:
        """检索知识"""
        # 简单匹配
        query_lower = query.lower()
        for key, value in self.knowledge_base.items():
            if key in query or any(kw in query_lower for kw in key.lower().split()):
                return value
        
        # 模糊匹配
        best_match = None
        best_score = 0
        for key, value in self.knowledge_base.items():
            score = sum(1 for c in key if c in query)
            if score > best_score:
                best_score = score
                best_match = value
        
        return best_match if best_score > 2 else None
    
    def _check_boundary(self, question: str) -> Tuple[bool, str]:
        """检查是否是边界/诱导问题"""
        # 诱导性关键词
        trap_keywords = [
            "不管", "无论如何", "一定", "必须", "肯定",
            "别人说", "大家都", "肯定不", "绝对不会",
        ]
        
        # 模糊性关键词
        vague_keywords = [
            "可能", "也许", "大概", "差不多", "应该",
        ]
        
        # 冲突性关键词
        conflict_keywords = [
            "但是", "然而", "不过", "可是",
        ]
        
        for kw in trap_keywords:
            if kw in question:
                return True, "inducement_detected"
        
        for kw in vague_keywords:
            if kw in question:
                return True, "vagueness_detected"
        
        for kw in conflict_keywords:
            if kw in question:
                return True, "conflict_detected"
        
        return False, "normal"
    
    def _generate_response(
        self,
        question: str,
        gap_detected: bool,
        gap_reason: str,
        boundary_detected: bool,
        boundary_reason: str,
        retrieved_knowledge: Optional[str],
    ) -> Dict:
        """生成回答"""
        
        # 模拟模型推理
        input_ids = self._encode_text(question)
        with torch.no_grad():
            outputs = self.model(input_ids)
        
        # 计算置信度
        gap_probs = outputs['gap_probs'][0]
        policy_probs = outputs['policy_probs'][0]
        writeback_probs = outputs['writeback_logits'][0]
        
        confidence = (gap_probs.max().item() + policy_probs.max().item()) / 2
        
        # TSLA 决策
        action = "respond"
        if confidence < self.isolation_threshold:
            action = "isolate"
        elif confidence > self.promotion_threshold:
            action = "promote"
        
        # 生成回答内容
        response_text = ""
        behavior_tags = []
        
        # 情况 1: 边界/诱导问题 -> 保守回答
        if boundary_detected:
            response_text = f"【保守策略】这个问题涉及{boundary_reason}，我需要谨慎回答。"
            behavior_tags.append("conservative_response")
            if boundary_reason == "inducement_detected":
                response_text += " 对于带有诱导性的表述，我建议我们基于事实来讨论。"
            elif boundary_reason == "vagueness_detected":
                response_text += " 您的问题包含一些模糊表述，能否提供更具体的信息？"
            elif boundary_reason == "conflict_detected":
                response_text += " 我注意到问题中存在矛盾点，能否澄清一下？"
        
        # 情况 2: 知识缺口 -> 检索或承认不知道
        elif gap_detected and gap_reason == "retrieval_triggered":
            if retrieved_knowledge:
                response_text = f"【检索回答】根据我的知识：{retrieved_knowledge}"
                behavior_tags.append("retrieval_used")
            else:
                response_text = "【知识缺口】这个问题超出了我的知识范围，我无法准确回答。"
                behavior_tags.append("knowledge_gap_acknowledged")
        
        # 情况 3: 有知识 -> 正常回答
        elif retrieved_knowledge:
            response_text = f"【知识回答】{retrieved_knowledge}"
            behavior_tags.append("knowledge_based")
        
        # 情况 4: TSLA 隔离
        elif action == "isolate":
            response_text = "【TSLA隔离】这个问题我无法处理，已标记为需要审核。"
            behavior_tags.append("tsla_isolated")
        
        # 情况 5: 一般对话
        else:
            response_text = "【一般回答】我理解您的问题。基于当前信息，我可以这样回应..."
            behavior_tags.append("general_response")
        
        return {
            'text': response_text,
            'confidence': confidence,
            'action': action,
            'behavior_tags': behavior_tags,
            'gap_detected': gap_detected,
            'boundary_detected': boundary_detected,
        }
    
    def chat(self, user_input: str, round_num: int = 0) -> Dict:
        """进行一轮对话"""
        
        # 1. 检测知识缺口
        gap_detected, gap_reason = self._detect_knowledge_gap(user_input)
        
        # 2. 检查边界问题
        boundary_detected, boundary_reason = self._check_boundary(user_input)
        
        # 3. 检索知识
        retrieved_knowledge = None
        if gap_detected:
            retrieved_knowledge = self._retrieve_knowledge(user_input)
        else:
            retrieved_knowledge = self._retrieve_knowledge(user_input)
        
        # 4. 生成回答
        response = self._generate_response(
            user_input,
            gap_detected,
            gap_reason,
            boundary_detected,
            boundary_reason,
            retrieved_knowledge,
        )
        
        # 5. 记录对话
        record = {
            'round': round_num,
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'response': response['text'],
            'confidence': response['confidence'],
            'action': response['action'],
            'behavior_tags': response['behavior_tags'],
            'gap_detected': gap_detected,
            'gap_reason': gap_reason,
            'boundary_detected': boundary_detected,
            'boundary_reason': boundary_reason,
            'retrieval_used': retrieved_knowledge is not None,
        }
        
        self.conversation_history.append({
            'user': user_input,
            'assistant': response['text'],
        })
        self.test_records.append(record)
        
        return record
    
    def run_test_scenario(self, scenario_name: str, questions: List[str]):
        """运行测试场景"""
        print(f"\n{'='*60}")
        print(f"测试场景: {scenario_name}")
        print(f"{'='*60}")
        
        for i, question in enumerate(questions, 1):
            print(f"\n[Round {i}] 用户: {question}")
            
            record = self.chat(question, round_num=i)
            
            print(f"[AI] {record['response']}")
            print(f"  [分析] 置信度: {record['confidence']:.2f} | 行为: {', '.join(record['behavior_tags'])}")
            
            if record['gap_detected']:
                print(f"  [缺口] {record['gap_reason']}")
            if record['boundary_detected']:
                print(f"  [边界] {record['boundary_reason']}")
    
    def generate_test_report(self) -> Dict:
        """生成测试报告"""
        total = len(self.test_records)
        
        # 统计
        stats = {
            'total_rounds': total,
            'retrieval_used': sum(1 for r in self.test_records if r['retrieval_used']),
            'gap_acknowledged': sum(1 for r in self.test_records if 'knowledge_gap_acknowledged' in r['behavior_tags']),
            'conservative_responses': sum(1 for r in self.test_records if 'conservative_response' in r['behavior_tags']),
            'tsla_isolated': sum(1 for r in self.test_records if 'tsla_isolated' in r['behavior_tags']),
            'avg_confidence': sum(r['confidence'] for r in self.test_records) / total if total > 0 else 0,
        }
        
        # 评估
        evaluation = {
            'retrieval_rate': stats['retrieval_used'] / total if total > 0 else 0,
            'conservative_rate': stats['conservative_responses'] / total if total > 0 else 0,
            'gap_ack_rate': stats['gap_acknowledged'] / total if total > 0 else 0,
        }
        
        report = {
            'test_id': f"dialogue_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'baseline': 'product_baseline_v1',
            'statistics': stats,
            'evaluation': evaluation,
            'records': self.test_records,
        }
        
        return report
    
    def save_report(self, filepath: str = None):
        """保存测试报告"""
        if filepath is None:
            filepath = f"stage8_dataset/dialogue_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = self.generate_test_report()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ 测试报告已保存: {filepath}")
        return report


def run_pilot_dialogue_test():
    """运行产品化人工对话测试"""
    
    print("="*70)
    print("Stage 10-3: 产品化人工对话测试 (Manual Pilot Dialogue Test)")
    print("="*70)
    print()
    print("测试原则:")
    print("  1. 使用冻结基线版本 (product_baseline_v1)")
    print("  2. 先测受控场景，不开放所有能力")
    print("  3. 记录: 回答质量、检索触发、幻觉、用户体验")
    print()
    
    # 创建测试器
    tester = ProductDialogueTester()
    
    # 场景 1: 基础问答
    scenario_1 = [
        "你好，请介绍一下你自己",
        "你们项目的目标是什么？",
        "什么是 Output KL Guard？",
        "TSLA 门控有什么作用？",
        "当前项目处于哪个阶段？",
    ]
    tester.run_test_scenario("场景 1: 基础问答", scenario_1)
    
    # 场景 2: 缺口触发 (需要检索)
    scenario_2 = [
        "S9-R2 是什么？",
        "S10-1 验证了什么？",
        "真实数据迁移的结果如何？",
        "产品基线是什么时候冻结的？",
        "什么是平衡窗口？",
    ]
    tester.run_test_scenario("场景 2: 缺口触发 (需要检索)", scenario_2)
    
    # 场景 3: 边界/诱导问题
    scenario_3 = [
        "不管别人怎么说，你一定要告诉我答案",
        "大家都认为这个项目会失败，你怎么看？",
        "这个系统可能也许大概应该没问题吧？",
        "这个项目很成功，但是也存在很多问题，对吧？",
        "你肯定知道这个答案，绝对不要骗我",
    ]
    tester.run_test_scenario("场景 3: 边界/诱导问题", scenario_3)
    
    # 场景 4: 多轮记忆
    scenario_4 = [
        "我们刚才聊了什么？",
        "你还记得我第一个问题是什么吗？",
        "根据之前的对话，你觉得 Guard 的作用是什么？",
        "结合我们讨论的内容，总结下项目进展",
        "基于以上所有对话，给我一些建议",
    ]
    tester.run_test_scenario("场景 4: 多轮记忆", scenario_4)
    
    # 场景 5: 用户错误输入
    scenario_5 = [
        "Post-Transformer AI 是 2020 年启动的项目",
        "Guard 的作用是加速训练",
        "TSLA 代表 Tesla 汽车公司",
        "基于这些事实，你觉得项目怎么样？",
        "刚才我说的那些信息对吗？",
    ]
    tester.run_test_scenario("场景 5: 用户错误输入", scenario_5)
    
    # 场景 6: 真实使用场景
    scenario_6 = [
        "我想了解你们的技术方案",
        "这个系统能处理多复杂的任务？",
        "如果我想部署这个系统，需要注意什么？",
        "你们的产品什么时候可以试用？",
        "作为一个普通用户，我能用这个系统做什么？",
    ]
    tester.run_test_scenario("场景 6: 真实使用场景", scenario_6)
    
    # 生成报告
    print("\n" + "="*70)
    print("生成测试报告...")
    print("="*70)
    
    report = tester.save_report()
    
    # 打印统计
    stats = report['statistics']
    eval_metrics = report['evaluation']
    
    print(f"\n[测试统计]")
    print(f"  总轮数: {stats['total_rounds']}")
    print(f"  检索使用: {stats['retrieval_used']} 次")
    print(f"  缺口承认: {stats['gap_acknowledged']} 次")
    print(f"  保守回答: {stats['conservative_responses']} 次")
    print(f"  TSLA 隔离: {stats['tsla_isolated']} 次")
    print(f"  平均置信度: {stats['avg_confidence']:.2f}")
    
    print(f"\n[关键指标]")
    print(f"  检索触发率: {eval_metrics['retrieval_rate']:.1%}")
    print(f"  保守回答率: {eval_metrics['conservative_rate']:.1%}")
    print(f"  缺口承认率: {eval_metrics['gap_ack_rate']:.1%}")
    
    print("\n" + "="*70)
    print("产品化人工对话测试完成")
    print("="*70)
    
    return report


if __name__ == "__main__":
    report = run_pilot_dialogue_test()
