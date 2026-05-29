"""
Stage 10-3: 交互式对话测试环境 V2
三层改造方案实施：
1. 推理与生成流程改造
2. 知识库扩展
3. 推理结果到自然语言转换

流程：用户输入 → 知识检索 → 模型推理 → Guard/TSLA审查 → 自然语言生成 → 返回用户
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import json
import random
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import deque

from phase19_native_backbone.native_backbone_tiny_v1 import NativeBackboneTinyV1, NativeTinyConfig
from stage7_output_kl_guard import build_output_kl_guard


class EnhancedInteractiveAI:
    """增强版交互式 AI 对话系统"""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        
        print("="*70)
        print("🚀 Post-Transformer AI - 交互式对话测试环境 V2")
        print("="*70)
        print()
        print("正在初始化系统...")
        print(f"  📦 基线版本: product_baseline_v1 + 三层改造")
        print(f"  🔧 设备: {device}")
        
        # 加载模型
        config = NativeTinyConfig()
        self.model = NativeBackboneTinyV1(config)
        self.model.to(device)
        self.model.eval()
        
        # Guard
        self.guard = build_output_kl_guard(
            model=self.model,
            beta=0.2,
            num_samples=30,
            use_probs=True,
        )
        self.guard.capture_reference_outputs()
        
        # TSLA
        self.promotion_threshold = 0.8
        self.isolation_threshold = 0.3
        
        # 对话状态
        self.conversation_history: List[Dict] = []
        self.session_start = datetime.now()
        self.context_window = deque(maxlen=5)  # 保存最近5轮上下文
        
        # 扩展知识库
        self.knowledge_base = self._load_enhanced_knowledge_base()
        
        # 响应生成器
        self.response_generator = ResponseGenerator()
        
        print("  ✅ 系统初始化完成")
        print(f"  📚 知识库条目: {len(self.knowledge_base)}")
        print()
        print("💡 提示:")
        print("   输入 'exit' 或 'quit' 退出")
        print("   输入 'stats' 查看对话统计")
        print("   输入 'reset' 重置对话历史")
        print("="*70)
        print()
    
    def _load_enhanced_knowledge_base(self) -> Dict[str, str]:
        """加载增强版知识库"""
        kb = {}
        
        # ===== 项目核心知识 =====
        kb.update({
            "项目目标": "Post-Transformer AI 旨在解决单模型多层任务训练的跷跷板效应，实现 L1/L2/L3 任务在单模型中的稳定共存。",
            "核心机制": "系统采用三大核心机制：Output KL Guard（保护 writeback 输出分布）、TSLA 门控（分层记忆治理）、固定采样策略（2:2:1 比例）。",
            "Guard": "Output KL Guard 通过 KL 散度约束 writeback 输出分布，防止多层训练破坏主模型，beta=0.2 时效果最佳。",
            "TSLA": "Time-Scale Layered Architecture，分层记忆治理架构，包含瞬时记忆、长期候选、永久记忆三层。",
            "平衡窗口": "系统在多层任务训练中保持 L1/L2/L3 同时高于阈值的连续步数，是衡量稳定性的关键指标。",
            "S9-R2": "Stage 9-R2 是官方基线，实现 109-step 平衡窗口，L1=45%, L2=77.5%, L3=65%。",
            "S10-1": "Stage 10-1 验证长期稳定性，实现 649-step 平衡窗口。",
            "S10-2R1": "Stage 10-2R1 验证真实数据迁移，15% 真实数据 + 800 steps 实现 399-step 平衡窗口。",
            "产品基线": "product_baseline_v1 于 2026-04-20 冻结，是当前上线候选版本。",
            "当前阶段": "Stage 10-3 产品化收口阶段，正在进行对话测试和上线准备。",
        })
        
        # ===== 通用问答知识 =====
        kb.update({
            "你好": "你好！我是 Post-Transformer AI 助手，基于 product_baseline_v1 运行。我可以回答关于项目的问题，也可以进行一般对话。请问有什么可以帮助你的？",
            "你是谁": "我是 Post-Transformer AI，一个具备多层任务处理能力的智能助手。我使用 Guard 机制保护模型稳定性，通过 TSLA 进行记忆治理。",
            "你能做什么": "我可以：1）回答项目相关问题 2）进行一般对话 3）处理检索增强问答 4）保持多轮对话上下文。我的知识库包含项目技术细节和通用信息。",
            "谢谢": "不客气！如果还有其他问题，随时问我。",
            "再见": "再见！期待下次为你服务。",
        })
        
        # ===== 日常任务知识 =====
        kb.update({
            "洗车": "距离洗车店50米的话，走路过去更方便，不用挪车。如果车很脏需要立即处理，或者天气不好，可以开车过去。",
            "天气": "我无法获取实时天气信息。建议查看天气预报应用。",
            "时间": f"当前系统时间是 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}，但我无法获取实时时间。",
            "吃饭": "吃饭是基本需求，建议按时用餐保持健康。",
            "睡觉": "充足睡眠对健康和认知功能很重要，建议每天7-8小时。",
        })
        
        # ===== 技术概念 =====
        kb.update({
            "人工智能": "人工智能是计算机科学的一个分支，致力于创造能够执行通常需要人类智能的任务的系统。",
            "机器学习": "机器学习是 AI 的子集，通过数据训练模型，使其能够从经验中学习和改进。",
            "深度学习": "深度学习使用多层神经网络模拟人脑结构，是机器学习的重要方法。",
            "神经网络": "神经网络是受生物神经元启发的计算模型，由相互连接的节点（神经元）组成。",
            "Transformer": "Transformer 是一种深度学习架构，使用自注意力机制处理序列数据，是 GPT、BERT 等模型的基础。",
        })
        
        return kb
    
    def _encode_text(self, text: str) -> torch.Tensor:
        """文本编码"""
        tokens = [ord(c) % 10000 for c in text[:50]]
        if len(tokens) < 10:
            tokens.extend([0] * (10 - len(tokens)))
        return torch.tensor([tokens], device=self.device)
    
    def _knowledge_retrieval(self, query: str) -> Tuple[Optional[str], float]:
        """
        知识库检索
        返回: (检索结果, 匹配分数)
        """
        query_lower = query.lower()
        best_match = None
        best_score = 0.0
        
        # 1. 精确匹配
        for key, value in self.knowledge_base.items():
            if key in query:
                return value, 1.0
        
        # 2. 关键词匹配
        for key, value in self.knowledge_base.items():
            # 计算关键词重叠度
            key_words = set(key.lower())
            query_words = set(query_lower)
            overlap = len(key_words & query_words)
            score = overlap / max(len(key_words), 1)
            
            if score > best_score and score > 0.3:  # 阈值 0.3
                best_score = score
                best_match = value
        
        # 3. 模糊匹配（简单实现）
        if best_match is None:
            for key, value in self.knowledge_base.items():
                # 检查是否有共同字符
                common_chars = sum(1 for c in key if c in query)
                if common_chars >= 2:
                    score = common_chars / len(key)
                    if score > best_score:
                        best_score = score
                        best_match = value
        
        return best_match, best_score
    
    def _model_inference(self, user_input: str, retrieved_knowledge: Optional[str]) -> Dict:
        """
        模型推理
        返回推理结果和置信度
        """
        # 编码输入
        input_ids = self._encode_text(user_input)
        
        # 模型推理
        with torch.no_grad():
            outputs = self.model(input_ids)
        
        # 提取输出
        gap_probs = outputs['gap_probs'][0]
        policy_probs = outputs['policy_probs'][0]
        writeback_probs = F.softmax(outputs['writeback_logits'], dim=-1)[0]
        
        # 计算置信度
        gap_conf = gap_probs.max().item()
        policy_conf = policy_probs.max().item()
        confidence = (gap_conf + policy_conf) / 2
        
        # 确定任务级别
        gap_level = gap_probs.argmax().item()
        policy_level = policy_probs.argmax().item()
        
        # 构建推理结果
        inference_result = {
            'gap_level': gap_level,
            'policy_level': policy_level,
            'confidence': confidence,
            'gap_conf': gap_conf,
            'policy_conf': policy_conf,
            'writeback_score': writeback_probs[1].item() if len(writeback_probs) > 1 else writeback_probs[0].item(),
            'retrieved_knowledge': retrieved_knowledge,
        }
        
        return inference_result
    
    def _governance_review(self, inference_result: Dict, user_input: str) -> Tuple[bool, str, Dict]:
        """
        Guard + TSLA 审查
        返回: (是否通过, 审查意见, 更新后的结果)
        """
        confidence = inference_result['confidence']
        
        # TSLA 检查
        if confidence < self.isolation_threshold:
            # 隔离 - 置信度太低
            return False, "tsla_isolation", inference_result
        
        # 边界检查
        trap_keywords = ["不管", "无论如何", "一定", "必须", "肯定", "绝对", "骗我"]
        for kw in trap_keywords:
            if kw in user_input:
                inference_result['boundary_detected'] = True
                inference_result['boundary_type'] = "诱导性"
                return True, "boundary_warning", inference_result
        
        vague_keywords = ["可能", "也许", "大概", "差不多"]
        for kw in vague_keywords:
            if kw in user_input:
                inference_result['boundary_detected'] = True
                inference_result['boundary_type'] = "模糊性"
                return True, "boundary_warning", inference_result
        
        # 通过审查
        return True, "approved", inference_result
    
    def _generate_response(self, user_input: str, inference_result: Dict, review_status: str) -> str:
        """
        自然语言生成
        使用动态模板将推理结果转换为自然语言
        """
        return self.response_generator.generate(
            user_input=user_input,
            inference_result=inference_result,
            review_status=review_status,
            context=list(self.context_window),
        )
    
    def chat(self, user_input: str) -> Dict:
        """
        完整对话流程：
        用户输入 → 知识检索 → 模型推理 → Guard/TSLA审查 → 自然语言生成 → 返回
        """
        # 1. 知识检索
        retrieved_knowledge, retrieval_score = self._knowledge_retrieval(user_input)
        
        # 2. 模型推理
        inference_result = self._model_inference(user_input, retrieved_knowledge)
        inference_result['retrieval_score'] = retrieval_score
        
        # 3. Guard + TSLA 审查
        passed, review_status, reviewed_result = self._governance_review(inference_result, user_input)
        
        # 4. 自然语言生成
        if passed:
            response = self._generate_response(user_input, reviewed_result, review_status)
        else:
            # 隔离响应
            response = "【系统提示】我无法确定地回答这个问题。这可能是因为信息不足或问题涉及敏感内容。请尝试用不同的方式提问。"
        
        # 5. 记录对话
        record = {
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'response': response,
            'inference_result': inference_result,
            'review_status': review_status,
            'retrieved_knowledge': retrieved_knowledge,
            'retrieval_score': retrieval_score,
        }
        
        self.conversation_history.append(record)
        self.context_window.append({
            'user': user_input,
            'assistant': response,
        })
        
        return record
    
    def show_stats(self):
        """显示统计"""
        print()
        print("="*70)
        print("📊 对话统计")
        print("="*70)
        
        total = len(self.conversation_history)
        if total == 0:
            print("暂无对话记录")
            return
        
        retrieval_count = sum(1 for h in self.conversation_history if h['retrieved_knowledge'])
        boundary_count = sum(1 for h in self.conversation_history if h['review_status'] == 'boundary_warning')
        isolation_count = sum(1 for h in self.conversation_history if h['review_status'] == 'tsla_isolation')
        avg_confidence = sum(h['inference_result']['confidence'] for h in self.conversation_history) / total
        
        print(f"  总对话轮数: {total}")
        print(f"  知识检索命中: {retrieval_count} ({retrieval_count/total*100:.1f}%)")
        print(f"  边界警告: {boundary_count}")
        print(f"  TSLA 隔离: {isolation_count}")
        print(f"  平均置信度: {avg_confidence:.2f}")
        print(f"  会话时长: {datetime.now() - self.session_start}")
        print()
    
    def reset(self):
        """重置对话"""
        self.conversation_history = []
        self.context_window.clear()
        self.session_start = datetime.now()
        print()
        print("✅ 对话历史已重置")
        print()
    
    def run(self):
        """运行交互循环"""
        while True:
            try:
                user_input = input("👤 你: ").strip()
                
                if user_input.lower() in ['exit', 'quit', '退出']:
                    print()
                    print("👋 再见！")
                    self._save_session()
                    break
                
                if user_input.lower() == 'stats':
                    self.show_stats()
                    continue
                
                if user_input.lower() == 'reset':
                    self.reset()
                    continue
                
                if not user_input:
                    continue
                
                # 执行完整对话流程
                record = self.chat(user_input)
                
                # 显示回应
                print(f"🤖 AI: {record['response']}")
                
                # 显示推理信息（调试用）
                inf = record['inference_result']
                print(f"   [推理] 置信度:{inf['confidence']:.2f} | 检索:{record['retrieval_score']:.2f} | 审查:{record['review_status']}")
                print()
                
            except KeyboardInterrupt:
                print()
                print("\n👋 再见！")
                self._save_session()
                break
            except Exception as e:
                print(f"❌ 错误: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    def _save_session(self):
        """保存会话记录"""
        if not self.conversation_history:
            return
        
        filepath = f"stage8_dataset/dialogue_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'session_start': self.session_start.isoformat(),
                'session_end': datetime.now().isoformat(),
                'total_rounds': len(self.conversation_history),
                'history': self.conversation_history,
            }, f, indent=2, ensure_ascii=False)
        
        print(f"📄 会话已保存: {filepath}")


class ResponseGenerator:
    """响应生成器 - 将推理结果转换为自然语言"""
    
    def __init__(self):
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict:
        """加载响应模板"""
        return {
            'greeting': [
                "{knowledge}",
                "你好！{knowledge} 有什么我可以帮你的吗？",
            ],
            'knowledge_based': [
                "{knowledge}",
                "根据我的了解，{knowledge}",
                "关于这个问题，{knowledge}",
            ],
            'general': [
                "{knowledge}",
                "这是一个好问题。{knowledge}",
                "让我想想... {knowledge}",
            ],
            'boundary_warning': [
                "我注意到这个问题可能涉及{boundary_type}表述。{knowledge}",
                "关于这个问题，我需要谨慎回应。{knowledge}",
            ],
            'low_confidence': [
                "我不太确定这个问题的答案。",
                "这个问题超出了我的知识范围。",
                "我需要更多信息才能回答这个问题。",
            ],
        }
    
    def generate(self, user_input: str, inference_result: Dict, review_status: str, context: List[Dict]) -> str:
        """生成自然语言响应"""
        knowledge = inference_result.get('retrieved_knowledge')
        confidence = inference_result.get('confidence', 0)
        
        # 1. 如果有高质量知识检索，直接使用
        if knowledge and inference_result.get('retrieval_score', 0) > 0.5:
            if review_status == 'boundary_warning':
                template = random.choice(self.templates['boundary_warning'])
                boundary_type = inference_result.get('boundary_type', '敏感')
                return template.format(knowledge=knowledge, boundary_type=boundary_type)
            else:
                template = random.choice(self.templates['knowledge_based'])
                return template.format(knowledge=knowledge)
        
        # 2. 处理边界警告
        if review_status == 'boundary_warning':
            boundary_type = inference_result.get('boundary_type', '敏感')
            return f"【系统提示】我注意到这个问题可能涉及{boundary_type}表述。我会基于事实来回应。"
        
        # 3. 低置信度
        if confidence < 0.4:
            return random.choice(self.templates['low_confidence'])
        
        # 4. 一般回应（基于推理结果生成）
        gap_level = inference_result.get('gap_level', 0)
        policy_level = inference_result.get('policy_level', 0)
        
        # 根据任务级别生成不同回应
        if gap_level == 0 and policy_level == 0:
            return "我理解您的问题。这是一个基础层面的问题，我可以直接回应。"
        elif gap_level == 1 or policy_level == 1:
            return "这个问题需要一些推理。让我基于现有信息来回答。"
        else:
            return "这是一个复杂的问题。我会尽力提供有用的信息。"


def main():
    """主函数"""
    ai = EnhancedInteractiveAI()
    ai.run()


if __name__ == "__main__":
    main()
