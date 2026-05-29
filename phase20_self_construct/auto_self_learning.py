"""
自动自学习系统 - 项目完成后自主成长

核心能力:
1. 自动发现知识缺口
2. 主动触发检索和学习
3. 生成候选知识并自我审查
4. 通过TSLA治理链晋升有效知识
5. 持续自我改进，无需人工干预

运行模式:
- 主动探索模式: 系统自动生成问题并学习
- 对话反思模式: 从用户对话中发现不足并学习
- 知识补全模式: 识别知识库缺口并主动填补
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import json
import random
import time
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import deque
import requests

from phase19_native_backbone.native_backbone_tiny_v1 import NativeBackboneTinyV1, NativeTinyConfig


# DeepSeek API 配置
DEEPSEEK_API_KEY = "sk-2296148b16f54ab0be255e98a911c9fe"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"


@dataclass
class LearningEpisode:
    """学习片段 - 记录一次完整自学习过程"""
    episode_id: str
    timestamp: str
    trigger_type: str  # 'gap_detected', 'user_feedback', 'exploration', 'conflict'
    
    # 问题与上下文
    query: str
    context: str = ""
    
    # 缺口识别
    gap_detected: bool = False
    gap_type: str = ""
    gap_confidence: float = 0.0
    
    # 检索过程
    retrieval_triggered: bool = False
    retrieval_results: List[Dict] = field(default_factory=list)
    
    # 候选生成
    candidate_generated: bool = False
    candidate_knowledge: str = ""
    candidate_confidence: float = 0.0
    
    # 教师指导 (Phase C逐步降低依赖)
    teacher_used: bool = False
    teacher_answer: str = ""
    teacher_score: float = 0.0
    
    # TSLA审查
    tsla_action: str = ""
    tsla_confidence: float = 0.0
    
    # 学习结果
    learned: bool = False
    knowledge_promoted: bool = False
    memory_location: str = ""  # 'instant', 'review', 'long_term', 'isolated'
    
    # 验证
    validation_score: float = 0.0
    used_in_response: bool = False


@dataclass
class AutoLearningMetrics:
    """自学习指标"""
    total_episodes: int = 0
    successful_learnings: int = 0
    promoted_to_long_term: int = 0
    isolated_count: int = 0
    
    # 触发类型分布
    gap_triggered: int = 0
    feedback_triggered: int = 0
    exploration_triggered: int = 0
    
    # 教师依赖度 (目标: 逐渐降低)
    teacher_dependency_ratio: float = 1.0
    
    # 自主能力指标 (目标: 逐渐升高)
    self_retrieval_rate: float = 0.0
    self_construction_rate: float = 0.0
    tsla_consistency: float = 0.0
    
    # 知识库增长
    kb_size_start: int = 0
    kb_size_current: int = 0
    
    def to_dict(self) -> Dict:
        return asdict(self)


class KnowledgeBaseManager:
    """知识库管理器 - 支持分层存储"""
    
    def __init__(self):
        # 四层知识存储
        self.instant_layer: Dict[str, Any] = {}      # 瞬时层 - 临时知识
        self.review_zone: Dict[str, Any] = {}         # 受审区 - 待验证知识
        self.long_term: Dict[str, Any] = {}           # 长期记忆 - 已验证知识
        self.isolated: Dict[str, Any] = {}            # 隔离区 - 错误/有害知识
        
        # 知识元数据
        self.knowledge_meta: Dict[str, Dict] = {}
        
        self._load_base_knowledge()
    
    def _load_base_knowledge(self):
        """加载基础知识"""
        base_knowledge = {
            # AI基础
            "人工智能": "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
            "机器学习": "机器学习是AI的子领域，让计算机从数据中学习规律，无需明确编程。",
            "深度学习": "深度学习使用多层神经网络学习数据的层次化表示。",
            "神经网络": "神经网络是受生物神经元启发的计算模型，能够学习复杂模式。",
            
            # 项目核心
            "Guard机制": "Guard通过KL散度约束writeback输出分布，防止知识写入灾难性漂移。",
            "TSLA": "TSLA(三态学习架构)管理知识的晋升、隔离和回流。",
            "自学习": "自学习让模型主动发现缺口、检索、构建、审查、晋升知识。",
        }
        
        for key, value in base_knowledge.items():
            self.long_term[key] = {
                'content': value,
                'source': 'base',
                'verified': True,
                'created_at': datetime.now().isoformat(),
            }
    
    def add_candidate(self, key: str, content: str, source: str = "self_constructed") -> str:
        """
        添加候选知识到受审区
        返回: 知识ID
        """
        knowledge_id = f"k_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
        
        self.review_zone[knowledge_id] = {
            'key': key,
            'content': content,
            'source': source,
            'created_at': datetime.now().isoformat(),
            'review_count': 0,
            'validation_score': 0.0,
        }
        
        self.knowledge_meta[knowledge_id] = {
            'location': 'review_zone',
            'key': key,
            'added_at': datetime.now().isoformat(),
        }
        
        return knowledge_id
    
    def promote(self, knowledge_id: str) -> bool:
        """将知识从受审区晋升到长期记忆"""
        if knowledge_id not in self.review_zone:
            return False
        
        knowledge = self.review_zone.pop(knowledge_id)
        knowledge['promoted_at'] = datetime.now().isoformat()
        knowledge['verified'] = True
        
        # 使用key作为长期记忆索引
        key = knowledge['key']
        self.long_term[key] = knowledge
        
        self.knowledge_meta[knowledge_id]['location'] = 'long_term'
        
        return True
    
    def isolate(self, knowledge_id: str, reason: str = "") -> bool:
        """隔离知识"""
        # 从受审区隔离
        if knowledge_id in self.review_zone:
            knowledge = self.review_zone.pop(knowledge_id)
            knowledge['isolated_at'] = datetime.now().isoformat()
            knowledge['isolation_reason'] = reason
            self.isolated[knowledge_id] = knowledge
            self.knowledge_meta[knowledge_id]['location'] = 'isolated'
            return True
        
        return False
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """搜索知识库"""
        results = []
        query_lower = query.lower()
        
        # 搜索长期记忆
        for key, value in self.long_term.items():
            score = 0.0
            if key.lower() in query_lower:
                score = 0.95
            elif any(word in query_lower for word in key.lower().split()):
                score = 0.7
            
            if score > 0.5:
                results.append({
                    'id': key,
                    'key': key,
                    'content': value.get('content', ''),
                    'score': score,
                    'source': value.get('source', 'unknown'),
                    'verified': value.get('verified', False),
                })
        
        # 搜索受审区
        for kid, value in self.review_zone.items():
            key = value.get('key', '')
            score = 0.0
            if key.lower() in query_lower:
                score = 0.8
            elif any(word in query_lower for word in key.lower().split()):
                score = 0.6
            
            if score > 0.5:
                results.append({
                    'id': kid,
                    'key': key,
                    'content': value.get('content', ''),
                    'score': score,
                    'source': value.get('source', 'unknown'),
                    'verified': False,
                })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def get_stats(self) -> Dict:
        """获取知识库统计"""
        return {
            'instant_layer': len(self.instant_layer),
            'review_zone': len(self.review_zone),
            'long_term': len(self.long_term),
            'isolated': len(self.isolated),
            'total': len(self.instant_layer) + len(self.review_zone) + len(self.long_term) + len(self.isolated),
        }
    
    def export_long_term(self, filepath: str):
        """导出长期记忆"""
        export_data = {}
        for key, value in self.long_term.items():
            export_data[key] = {
                'content': value.get('content', ''),
                'source': value.get('source', ''),
                'verified': value.get('verified', False),
            }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)


class GapDetector:
    """缺口检测器 - 自动发现知识缺口"""
    
    def __init__(self, kb_manager: KnowledgeBaseManager):
        self.kb = kb_manager
        self.confidence_threshold = 0.6
    
    def detect(self, query: str) -> Tuple[bool, str, float]:
        """
        检测是否存在知识缺口
        返回: (是否有缺口, 缺口类型, 置信度)
        """
        # 搜索现有知识
        results = self.kb.search(query, top_k=3)
        
        if not results:
            # 完全无相关知识
            return True, "complete_gap", 0.9
        
        best_score = results[0]['score']
        
        if best_score > 0.9:
            # 知识充足
            return False, "none", best_score
        elif best_score > 0.7:
            # 部分缺口
            return True, "partial_gap", 1 - best_score
        else:
            # 显著缺口
            return True, "significant_gap", 1 - best_score
    
    def generate_exploration_questions(self, num_questions: int = 5) -> List[str]:
        """生成探索性问题 - 主动发现缺口"""
        # 基于当前知识库，生成相关问题
        exploration_templates = [
            "{}的应用场景有哪些？",
            "{}和{}有什么区别？",
            "如何实现{}？",
            "{}的优缺点是什么？",
            "什么是{}的最佳实践？",
            "{}的未来发展趋势是什么？",
            "如何解决{}中的常见问题？",
        ]
        
        # 获取知识库中的关键概念
        concepts = list(self.kb.long_term.keys())[:10]
        
        questions = []
        for _ in range(num_questions):
            if len(concepts) >= 2:
                template = random.choice(exploration_templates)
                if "{}和{}" in template:
                    c1, c2 = random.sample(concepts, 2)
                    questions.append(template.format(c1, c2))
                else:
                    c = random.choice(concepts)
                    questions.append(template.format(c))
        
        return questions


class TeacherClient:
    """教师客户端 - DeepSeek API"""
    
    def __init__(self, api_key: str = DEEPSEEK_API_KEY):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.call_count = 0
        self.enabled = True  # Phase C 会逐渐降低使用
    
    def query(self, question: str, context: str = "") -> Tuple[str, float]:
        """查询教师"""
        if not self.enabled:
            return "", 0.0
        
        try:
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "你是一个知识丰富的教师。请提供准确、简洁的答案。"},
                    {"role": "user", "content": f"{context}\n问题: {question}"}
                ],
                "temperature": 0.7,
                "max_tokens": 300,
            }
            
            response = requests.post(
                DEEPSEEK_API_URL,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            self.call_count += 1
            
            answer = result['choices'][0]['message']['content']
            
            # 简单评分
            score = 0.8 if len(answer) > 50 else 0.6
            
            return answer, score
            
        except Exception as e:
            print(f"[Teacher Error] {e}")
            return "", 0.0
    
    def set_dependency_level(self, level: float):
        """设置教师依赖度 (1.0=完全依赖, 0.0=完全不依赖)"""
        self.enabled = level > 0.1


class AutoSelfLearning:
    """自动自学习系统"""
    
    def __init__(self):
        # 初始化组件
        self.kb_manager = KnowledgeBaseManager()
        self.gap_detector = GapDetector(self.kb_manager)
        self.teacher = TeacherClient()
        
        # 学习历史
        self.episodes: deque = deque(maxlen=1000)
        self.metrics = AutoLearningMetrics()
        self.metrics.kb_size_start = self.kb_manager.get_stats()['total']
        
        # 运行状态
        self.running = False
        self.learning_thread = None
        
        # 配置
        self.config = {
            'exploration_interval': 60,  # 探索间隔(秒)
            'teacher_dependency': 1.0,    # 当前教师依赖度
            'min_confidence_for_promotion': 0.8,
            'max_episodes_per_session': 100,
        }
    
    def start(self):
        """启动自动学习"""
        self.running = True
        self.learning_thread = threading.Thread(target=self._learning_loop)
        self.learning_thread.daemon = True
        self.learning_thread.start()
        print("[AutoLearning] 自动自学习已启动")
    
    def stop(self):
        """停止自动学习"""
        self.running = False
        if self.learning_thread:
            self.learning_thread.join(timeout=5)
        print("[AutoLearning] 自动自学习已停止")
    
    def _learning_loop(self):
        """学习主循环"""
        episode_count = 0
        
        while self.running and episode_count < self.config['max_episodes_per_session']:
            # 1. 生成探索问题
            questions = self.gap_detector.generate_exploration_questions(num_questions=3)
            
            for question in questions:
                if not self.running:
                    break
                
                # 执行学习片段
                episode = self._learning_episode(question, trigger_type='exploration')
                self.episodes.append(episode)
                
                episode_count += 1
                self.metrics.total_episodes = episode_count
                
                # 更新统计
                if episode.learned:
                    self.metrics.successful_learnings += 1
                if episode.knowledge_promoted:
                    self.metrics.promoted_to_long_term += 1
                
                # 打印进度
                if episode_count % 10 == 0:
                    self._print_progress()
                
                # 间隔
                time.sleep(2)
            
            # 等待下一次探索
            time.sleep(self.config['exploration_interval'])
        
        print(f"\n[AutoLearning] 学习完成，共 {episode_count} 个片段")
    
    def _learning_episode(self, query: str, trigger_type: str = 'exploration') -> LearningEpisode:
        """执行一个学习片段"""
        episode = LearningEpisode(
            episode_id=f"ep_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}",
            timestamp=datetime.now().isoformat(),
            trigger_type=trigger_type,
            query=query,
        )
        
        # Step 1: 缺口检测
        has_gap, gap_type, gap_conf = self.gap_detector.detect(query)
        episode.gap_detected = has_gap
        episode.gap_type = gap_type
        episode.gap_confidence = gap_conf
        
        if not has_gap:
            # 无缺口，无需学习
            return episode
        
        # Step 2: 检索现有知识
        retrieval_results = self.kb_manager.search(query, top_k=3)
        episode.retrieval_triggered = True
        episode.retrieval_results = retrieval_results
        
        # Step 3: 生成候选知识
        candidate = self._generate_candidate(query, retrieval_results)
        episode.candidate_generated = True
        episode.candidate_knowledge = candidate
        episode.candidate_confidence = 0.6  # 初始置信度
        
        # Step 4: 教师指导 (根据依赖度决定是否使用)
        if random.random() < self.config['teacher_dependency']:
            teacher_answer, teacher_score = self.teacher.query(query)
            episode.teacher_used = True
            episode.teacher_answer = teacher_answer
            episode.teacher_score = teacher_score
            
            # 用教师答案改进候选
            if teacher_score > 0.7:
                episode.candidate_knowledge = teacher_answer
                episode.candidate_confidence = teacher_score
        
        # Step 5: TSLA审查
        tsla_action, tsla_conf = self._tsla_review(episode)
        episode.tsla_action = tsla_action
        episode.tsla_confidence = tsla_conf
        
        # Step 6: 执行学习
        if tsla_action == 'promote':
            # 晋升到长期记忆
            knowledge_id = self.kb_manager.add_candidate(
                key=query[:20],  # 简化key
                content=episode.candidate_knowledge,
                source='auto_learning'
            )
            self.kb_manager.promote(knowledge_id)
            episode.learned = True
            episode.knowledge_promoted = True
            episode.memory_location = 'long_term'
            
        elif tsla_action == 'review':
            # 进入受审区
            knowledge_id = self.kb_manager.add_candidate(
                key=query[:20],
                content=episode.candidate_knowledge,
                source='auto_learning'
            )
            episode.learned = True
            episode.memory_location = 'review_zone'
            
        elif tsla_action == 'isolate':
            # 隔离
            episode.memory_location = 'isolated'
            self.metrics.isolated_count += 1
        
        return episode
    
    def _generate_candidate(self, query: str, retrieval_results: List[Dict]) -> str:
        """生成候选知识"""
        if retrieval_results:
            # 基于检索结果整合
            base = retrieval_results[0]['content']
            return f"基于现有知识: {base[:100]}... [需要扩展]"
        else:
            # 全新知识
            return f"关于'{query}'的新知识待学习..."
    
    def _tsla_review(self, episode: LearningEpisode) -> Tuple[str, float]:
        """TSLA审查"""
        confidence = episode.candidate_confidence
        
        if confidence > self.config['min_confidence_for_promotion']:
            return 'promote', confidence
        elif confidence > 0.5:
            return 'review', confidence
        else:
            return 'isolate', 1 - confidence
    
    def _print_progress(self):
        """打印学习进度"""
        stats = self.kb_manager.get_stats()
        print(f"\n[Progress] 片段: {self.metrics.total_episodes} | "
              f"成功: {self.metrics.successful_learnings} | "
              f"晋升: {self.metrics.promoted_to_long_term} | "
              f"知识库: {stats}")
    
    def learn_from_user_feedback(self, query: str, feedback: str, rating: float):
        """从用户反馈中学习"""
        episode = self._learning_episode(query, trigger_type='user_feedback')
        episode.context = feedback
        episode.validation_score = rating
        
        self.episodes.append(episode)
        
        # 高评分知识优先晋升
        if rating > 0.8 and episode.knowledge_promoted:
            print(f"[Feedback] 高评分知识已学习: {query[:30]}...")
    
    def get_metrics(self) -> Dict:
        """获取学习指标"""
        stats = self.kb_manager.get_stats()
        self.metrics.kb_size_current = stats['total']
        
        # 计算自主能力指标
        total = self.metrics.total_episodes
        if total > 0:
            self.metrics.self_retrieval_rate = sum(1 for ep in self.episodes if ep.retrieval_triggered) / total
            self.metrics.self_construction_rate = sum(1 for ep in self.episodes if ep.candidate_generated) / total
            self.metrics.teacher_dependency_ratio = sum(1 for ep in self.episodes if ep.teacher_used) / total
        
        return self.metrics.to_dict()
    
    def export_knowledge(self, filepath: str):
        """导出学习到的知识"""
        self.kb_manager.export_long_term(filepath)
        print(f"[Export] 知识已导出: {filepath}")
    
    def reduce_teacher_dependency(self, target_level: float = 0.3):
        """降低教师依赖度 - Phase C"""
        print(f"[Phase C] 降低教师依赖: {self.config['teacher_dependency']:.1f} -> {target_level:.1f}")
        self.config['teacher_dependency'] = target_level
        self.teacher.set_dependency_level(target_level)


def main():
    """主函数 - 演示自动自学习"""
    print("="*70)
    print("🚀 自动自学习系统")
    print("="*70)
    print("系统将在后台自动:")
    print("  1. 发现知识缺口")
    print("  2. 检索现有知识")
    print("  3. 生成候选知识")
    print("  4. 教师指导(逐步降低依赖)")
    print("  5. TSLA审查与晋升")
    print("="*70)
    
    # 创建系统
    learner = AutoSelfLearning()
    
    # 启动自动学习
    learner.start()
    
    try:
        # 运行一段时间
        print("\n自动学习中... (按Ctrl+C停止)")
        time.sleep(300)  # 运行5分钟
        
        # 停止
        learner.stop()
        
        # 打印结果
        print("\n" + "="*70)
        print("📊 学习结果")
        print("="*70)
        
        metrics = learner.get_metrics()
        print(f"总学习片段: {metrics['total_episodes']}")
        print(f"成功学习: {metrics['successful_learnings']}")
        print(f"晋升长期记忆: {metrics['promoted_to_long_term']}")
        print(f"教师依赖度: {metrics['teacher_dependency_ratio']:.2%}")
        print(f"自主检索率: {metrics['self_retrieval_rate']:.2%}")
        print(f"知识库增长: {metrics['kb_size_start']} -> {metrics['kb_size_current']}")
        
        # 导出知识
        learner.export_knowledge("stage8_dataset/auto_learned_knowledge.json")
        
    except KeyboardInterrupt:
        print("\n\n用户中断")
        learner.stop()


if __name__ == "__main__":
    main()
