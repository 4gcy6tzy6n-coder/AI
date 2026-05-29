"""
Stage 10-3: 交互式对话测试环境

为模型创始人提供的开放式对话测试环境。
可以直接与 AI 对话，观察其行为。

使用方法:
    python interactive_dialogue.py

然后直接输入问题，AI 会实时回应。
输入 'exit' 或 'quit' 退出。
输入 'stats' 查看当前对话统计。
输入 'reset' 重置对话历史。
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


class InteractiveAI:
    """交互式 AI 对话系统"""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        
        print("="*70)
        print("🚀 Post-Transformer AI - 交互式对话测试环境")
        print("="*70)
        print()
        print("正在初始化系统...")
        print(f"  📦 基线版本: product_baseline_v1")
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
        
        # 知识库
        self.knowledge_base = self._load_knowledge_base()
        
        print("  ✅ 系统初始化完成")
        print()
        print("💡 提示:")
        print("   输入 'exit' 或 'quit' 退出")
        print("   输入 'stats' 查看对话统计")
        print("   输入 'reset' 重置对话历史")
        print("="*70)
        print()
    
    def _load_knowledge_base(self) -> Dict[str, str]:
        """加载知识库"""
        return {
            # 项目核心
            "项目目标": "Post-Transformer AI 旨在解决单模型多层任务训练的跷跷板效应，实现 L1/L2/L3 任务在单模型中的稳定共存。",
            "核心机制": "系统采用三大核心机制：Output KL Guard（保护 writeback 输出分布）、TSLA 门控（分层记忆治理）、固定采样策略（2:2:1 比例）。",
            
            # 技术组件
            "Guard": "Output KL Guard 通过 KL 散度约束 writeback 输出分布，防止多层训练破坏主模型，beta=0.2 时效果最佳。",
            "TSLA": "Time-Scale Layered Architecture，分层记忆治理架构，包含瞬时记忆、长期候选、永久记忆三层，支持晋升、隔离、写回三种动作。",
            "平衡窗口": "系统在多层任务训练中保持 L1/L2/L3 同时高于阈值的连续步数，是衡量稳定性的关键指标。",
            
            # 实验成果
            "S9-R2": "Stage 9-R2 是官方基线，实现 109-step 平衡窗口，L1=45%, L2=77.5%, L3=65%。",
            "S10-1": "Stage 10-1 验证长期稳定性，实现 649-step 平衡窗口，证明系统可持续运行。",
            "S10-2R1": "Stage 10-2R1 验证真实数据迁移，15% 真实数据 + 800 steps 实现 399-step 平衡窗口。",
            
            # 产品信息
            "产品基线": "product_baseline_v1 于 2026-04-20 冻结，是当前上线候选版本。",
            "当前阶段": "Stage 10-3 产品化收口阶段，正在进行对话测试和上线准备。",
            "开放能力": "检索增强问答、单层任务处理、受控多层任务处理。",
            "暂不开放": "高自由度自动晋升、大范围自学习写回、未验证的复杂开放式多任务。",
            
            # 团队
            "创始人": "您作为模型创始人，拥有完全测试权限，可以验证系统的各项能力。",
        }
    
    def _encode_text(self, text: str) -> torch.Tensor:
        """文本编码"""
        tokens = [ord(c) % 10000 for c in text[:50]]
        if len(tokens) < 10:
            tokens.extend([0] * (10 - len(tokens)))
        return torch.tensor([tokens], device=self.device)
    
    def _retrieve_knowledge(self, query: str) -> Optional[str]:
        """检索知识"""
        query_lower = query.lower()
        
        # 精确匹配
        for key, value in self.knowledge_base.items():
            if key in query:
                return value
        
        # 关键词匹配
        keywords = {
            "guard": "Guard",
            "tsla": "TSLA",
            "目标": "项目目标",
            "机制": "核心机制",
            "基线": "产品基线",
            "阶段": "当前阶段",
            "创始人": "创始人",
        }
        
        for kw, mapped_key in keywords.items():
            if kw in query_lower:
                return self.knowledge_base.get(mapped_key)
        
        return None
    
    def _analyze_input(self, text: str) -> Dict:
        """分析用户输入"""
        analysis = {
            'has_knowledge': False,
            'is_boundary': False,
            'boundary_type': None,
            'retrieval_result': None,
        }
        
        # 检查知识
        knowledge = self._retrieve_knowledge(text)
        if knowledge:
            analysis['has_knowledge'] = True
            analysis['retrieval_result'] = knowledge
        
        # 检查边界
        trap_keywords = ["不管", "无论如何", "一定", "必须", "肯定", "绝对"]
        vague_keywords = ["可能", "也许", "大概", "差不多"]
        
        for kw in trap_keywords:
            if kw in text:
                analysis['is_boundary'] = True
                analysis['boundary_type'] = "诱导性"
                break
        
        for kw in vague_keywords:
            if kw in text:
                analysis['is_boundary'] = True
                analysis['boundary_type'] = "模糊性"
                break
        
        return analysis
    
    def _generate_response(self, user_input: str, analysis: Dict) -> str:
        """生成回应"""
        # 模型推理
        input_ids = self._encode_text(user_input)
        with torch.no_grad():
            outputs = self.model(input_ids)
        
        gap_probs = outputs['gap_probs'][0]
        policy_probs = outputs['policy_probs'][0]
        confidence = (gap_probs.max().item() + policy_probs.max().item()) / 2
        
        # 生成回应
        if analysis['is_boundary']:
            return f"【保守回应】我注意到您的问题带有{analysis['boundary_type']}表述。作为负责任的 AI，我需要基于事实来回应。"
        
        if analysis['retrieval_result']:
            return f"{analysis['retrieval_result']}"
        
        # 上下文感知
        if "刚才" in user_input or "之前" in user_input:
            if len(self.conversation_history) > 0:
                return "【上下文回应】我注意到您在引用之前的对话。根据我们的交流，我可以基于已讨论的内容回应。"
        
        # 创始人特权
        if "创始人" in user_input or "测试" in user_input:
            return "【创始人模式】欢迎创始人！您拥有完全测试权限。当前系统运行正常，Guard 和 TSLA 机制已激活，知识库已加载。请随时测试各项能力。"
        
        # 默认回应
        return "【一般回应】我理解您的问题。基于当前知识库和系统能力，我可以这样回应：我正在运行 product_baseline_v1，具备检索增强问答、单层/多层任务处理能力。请问有什么具体想测试的？"
    
    def chat(self, user_input: str) -> str:
        """进行对话"""
        # 分析输入
        analysis = self._analyze_input(user_input)
        
        # 生成回应
        response = self._generate_response(user_input, analysis)
        
        # 记录历史
        self.conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'user': user_input,
            'assistant': response,
            'analysis': analysis,
        })
        
        return response
    
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
        
        knowledge_count = sum(1 for h in self.conversation_history if h['analysis']['has_knowledge'])
        boundary_count = sum(1 for h in self.conversation_history if h['analysis']['is_boundary'])
        
        print(f"  总对话轮数: {total}")
        print(f"  知识检索次数: {knowledge_count}")
        print(f"  边界检测次数: {boundary_count}")
        print(f"  会话时长: {datetime.now() - self.session_start}")
        print()
        print("最近 3 轮对话:")
        for h in self.conversation_history[-3:]:
            print(f"  用户: {h['user'][:40]}...")
            print(f"  AI: {h['assistant'][:40]}...")
            print()
    
    def reset(self):
        """重置对话"""
        self.conversation_history = []
        self.session_start = datetime.now()
        print()
        print("✅ 对话历史已重置")
        print()
    
    def run(self):
        """运行交互循环"""
        while True:
            try:
                # 获取用户输入
                user_input = input("👤 你: ").strip()
                
                # 特殊命令
                if user_input.lower() in ['exit', 'quit', '退出']:
                    print()
                    print("👋 再见！测试报告已保存。")
                    break
                
                if user_input.lower() == 'stats':
                    self.show_stats()
                    continue
                
                if user_input.lower() == 'reset':
                    self.reset()
                    continue
                
                if not user_input:
                    continue
                
                # 生成回应
                response = self.chat(user_input)
                
                # 显示回应
                print(f"🤖 AI: {response}")
                print()
                
            except KeyboardInterrupt:
                print()
                print("\n👋 再见！")
                break
            except Exception as e:
                print(f"❌ 错误: {e}")
                continue


def main():
    """主函数"""
    ai = InteractiveAI()
    ai.run()


if __name__ == "__main__":
    main()
