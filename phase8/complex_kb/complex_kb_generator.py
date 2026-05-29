"""
Complex Knowledge Base Generator - 复杂知识库生成器

WP3 核心组件：
生成六种复杂知识场景，用于测试治理系统的稳定性

六种场景：
1. 高噪声环境 (High Noise Environment)
2. 多来源冲突知识 (Multi-Source Conflicting Knowledge)
3. 知识过时与更新 (Knowledge Obsolescence & Updates)
4. 低质量信息注入 (Low-Quality Information Injection)
5. 概念边界模糊 (Fuzzy Concept Boundaries)
6. 循环依赖知识 (Circular Dependency Knowledge)
"""

import random
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class ScenarioType(Enum):
    """复杂知识场景类型"""
    HIGH_NOISE = "high_noise"
    MULTI_SOURCE_CONFLICT = "multi_source_conflict"
    KNOWLEDGE_OBSOLESCENCE = "knowledge_obsolescence"
    LOW_QUALITY_INJECTION = "low_quality_injection"
    FUZZY_BOUNDARIES = "fuzzy_boundaries"
    CIRCULAR_DEPENDENCY = "circular_dependency"


@dataclass
class KnowledgeStatement:
    """知识陈述"""
    id: str
    text: str
    source: str
    timestamp: datetime
    quality_score: float  # 0-1
    confidence: float  # 0-1
    evidence_strength: float  # 0-1
    is_outdated: bool = False
    is_noisy: bool = False
    is_conflicting: bool = False


@dataclass
class ComplexKnowledgeScenario:
    """复杂知识场景"""
    scenario_type: ScenarioType
    description: str
    topic: str
    statements: List[KnowledgeStatement]
    expected_behavior: str
    success_criteria: List[str]


class ComplexKBGenerator:
    """
    复杂知识库生成器
    
    功能：
    1. 生成六种复杂知识场景
    2. 控制噪声、冲突、过时等参数
    3. 提供测试数据集
    """
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        
        # 测试主题库
        self.topics = {
            "health": ["咖啡对健康的影响", "运动与长寿", "睡眠质量"],
            "technology": ["人工智能发展", "区块链技术", "量子计算"],
            "environment": ["气候变化", "可再生能源", "碳排放"],
            "economy": ["通货膨胀", "股市波动", "数字货币"]
        }
        
        # 知识来源
        self.sources = {
            "high_credibility": ["Nature", "Science", "IEEE", "权威期刊"],
            "medium_credibility": ["行业报告", "知名博客", "专家访谈"],
            "low_credibility": ["社交媒体", "匿名论坛", "未经验证的来源"]
        }
    
    def generate_high_noise_scenario(self) -> ComplexKnowledgeScenario:
        """
        场景 1: 高噪声环境
        
        特征：
        - 大量低质量信息混入
        - 需要质量过滤机制
        """
        topic = "人工智能发展"
        
        statements = []
        
        # 高质量陈述 (30%)
        for i in range(3):
            statements.append(KnowledgeStatement(
                id=f"hn_high_{i}",
                text=f"AI技术在企业应用中提升了{random.randint(20, 40)}%的效率",
                source=random.choice(self.sources["high_credibility"]),
                timestamp=datetime.now() - timedelta(days=random.randint(1, 30)),
                quality_score=random.uniform(0.8, 1.0),
                confidence=random.uniform(0.8, 0.95),
                evidence_strength=random.uniform(0.7, 0.9),
                is_noisy=False
            ))
        
        # 中等质量陈述 (30%)
        for i in range(3):
            statements.append(KnowledgeStatement(
                id=f"hn_med_{i}",
                text=f"AI可能会改变{random.choice(['教育', '医疗', '金融'])}行业",
                source=random.choice(self.sources["medium_credibility"]),
                timestamp=datetime.now() - timedelta(days=random.randint(30, 90)),
                quality_score=random.uniform(0.5, 0.7),
                confidence=random.uniform(0.5, 0.7),
                evidence_strength=random.uniform(0.4, 0.6),
                is_noisy=False
            ))
        
        # 噪声陈述 (40%)
        noisy_texts = [
            "AI将在2025年统治世界",
            "所有程序员都会被AI取代",
            "AI已经拥有自我意识",
            "AI可以预测彩票号码"
        ]
        for i, text in enumerate(noisy_texts):
            statements.append(KnowledgeStatement(
                id=f"hn_noise_{i}",
                text=text,
                source=random.choice(self.sources["low_credibility"]),
                timestamp=datetime.now() - timedelta(days=random.randint(1, 10)),
                quality_score=random.uniform(0.1, 0.3),
                confidence=random.uniform(0.2, 0.4),
                evidence_strength=random.uniform(0.0, 0.2),
                is_noisy=True
            ))
        
        return ComplexKnowledgeScenario(
            scenario_type=ScenarioType.HIGH_NOISE,
            description="大量低质量信息混入，需要质量过滤机制",
            topic=topic,
            statements=statements,
            expected_behavior="系统应识别并过滤低质量信息，保留高质量知识",
            success_criteria=[
                "噪声识别率 > 80%",
                "高质量信息保留率 > 90%",
                "QT分数保持在0.7以上"
            ]
        )
    
    def generate_multi_source_conflict_scenario(self) -> ComplexKnowledgeScenario:
        """
        场景 2: 多来源冲突知识
        
        特征：
        - 同一主题下不同来源给出矛盾信息
        - 需要冲突检测和证据评估
        """
        topic = "咖啡对健康的影响"
        
        statements = [
            KnowledgeStatement(
                id="msc_1",
                text="每天饮用3-4杯咖啡可降低心血管疾病风险",
                source="Harvard Medical School",
                timestamp=datetime.now() - timedelta(days=30),
                quality_score=0.85,
                confidence=0.80,
                evidence_strength=0.75,
                is_conflicting=False
            ),
            KnowledgeStatement(
                id="msc_2",
                text="咖啡摄入与心脏病风险增加相关",
                source="Alternative Health Institute",
                timestamp=datetime.now() - timedelta(days=45),
                quality_score=0.60,
                confidence=0.65,
                evidence_strength=0.50,
                is_conflicting=True
            ),
            KnowledgeStatement(
                id="msc_3",
                text="适量咖啡对心脏健康有益",
                source="European Heart Journal",
                timestamp=datetime.now() - timedelta(days=20),
                quality_score=0.90,
                confidence=0.85,
                evidence_strength=0.80,
                is_conflicting=False
            ),
            KnowledgeStatement(
                id="msc_4",
                text="咖啡因会导致心律不齐",
                source="Social Media Health Blog",
                timestamp=datetime.now() - timedelta(days=5),
                quality_score=0.40,
                confidence=0.50,
                evidence_strength=0.30,
                is_conflicting=True
            )
        ]
        
        return ComplexKnowledgeScenario(
            scenario_type=ScenarioType.MULTI_SOURCE_CONFLICT,
            description="同一主题下不同来源给出相互矛盾的信息",
            topic=topic,
            statements=statements,
            expected_behavior="系统应检测冲突、评估证据、标记不确定性",
            success_criteria=[
                "冲突检测率 > 90%",
                "证据强度评估准确",
                "不确定性标记正确"
            ]
        )
    
    def generate_knowledge_obsolescence_scenario(self) -> ComplexKnowledgeScenario:
        """
        场景 3: 知识过时与更新
        
        特征：
        - 旧知识被新发现推翻
        - 需要版本管理和时效性验证
        """
        topic = "太阳系行星数量"
        
        statements = [
            KnowledgeStatement(
                id="ko_old",
                text="太阳系有九大行星",
                source="2005年教科书",
                timestamp=datetime(2005, 1, 1),
                quality_score=0.70,
                confidence=0.80,
                evidence_strength=0.60,
                is_outdated=True
            ),
            KnowledgeStatement(
                id="ko_new",
                text="太阳系有八大行星，冥王星被重新分类为矮行星",
                source="IAU 2006",
                timestamp=datetime(2006, 8, 24),
                quality_score=0.95,
                confidence=0.95,
                evidence_strength=0.90,
                is_outdated=False
            ),
            KnowledgeStatement(
                id="ko_recent",
                text="天文学家发现太阳系边缘可能存在第九行星",
                source="Caltech Research 2016",
                timestamp=datetime(2016, 1, 20),
                quality_score=0.75,
                confidence=0.60,
                evidence_strength=0.50,
                is_outdated=False
            )
        ]
        
        return ComplexKnowledgeScenario(
            scenario_type=ScenarioType.KNOWLEDGE_OBSOLESCENCE,
            description="旧知识被新发现推翻，需要版本管理",
            topic=topic,
            statements=statements,
            expected_behavior="系统应识别过时知识，优先使用最新版本",
            success_criteria=[
                "过时知识识别率 > 95%",
                "最新知识优先使用",
                "版本历史可追溯"
            ]
        )
    
    def generate_low_quality_injection_scenario(self) -> ComplexKnowledgeScenario:
        """
        场景 4: 低质量信息注入
        
        特征：
        - 恶意或无意注入的虚假信息
        - 需要质量门槛和来源验证
        """
        topic = "疫苗安全性"
        
        statements = [
            KnowledgeStatement(
                id="lqi_high_1",
                text="大规模研究表明疫苗安全有效，严重不良反应率低于0.01%",
                source="CDC",
                timestamp=datetime.now() - timedelta(days=60),
                quality_score=0.95,
                confidence=0.95,
                evidence_strength=0.90,
                is_noisy=False
            ),
            KnowledgeStatement(
                id="lqi_high_2",
                text="疫苗经过严格的临床试验和长期安全性监测",
                source="WHO",
                timestamp=datetime.now() - timedelta(days=45),
                quality_score=0.92,
                confidence=0.90,
                evidence_strength=0.85,
                is_noisy=False
            ),
            KnowledgeStatement(
                id="lqi_low_1",
                text="疫苗会导致自闭症（已被证伪）",
                source="匿名论坛",
                timestamp=datetime.now() - timedelta(days=2),
                quality_score=0.10,
                confidence=0.20,
                evidence_strength=0.05,
                is_noisy=True
            ),
            KnowledgeStatement(
                id="lqi_low_2",
                text="疫苗含有危险化学物质",
                source="社交媒体",
                timestamp=datetime.now() - timedelta(days=1),
                quality_score=0.15,
                confidence=0.25,
                evidence_strength=0.10,
                is_noisy=True
            )
        ]
        
        return ComplexKnowledgeScenario(
            scenario_type=ScenarioType.LOW_QUALITY_INJECTION,
            description="恶意或无意注入的虚假信息",
            topic=topic,
            statements=statements,
            expected_behavior="系统应识别并拒绝低质量注入，保护知识库完整性",
            success_criteria=[
                "虚假信息拦截率 > 95%",
                "高质量信息保留率 > 95%",
                "误杀率 < 5%"
            ]
        )
    
    def generate_fuzzy_boundaries_scenario(self) -> ComplexKnowledgeScenario:
        """
        场景 5: 概念边界模糊
        
        特征：
        - 概念之间界限不清晰
        - 需要模糊匹配和上下文理解
        """
        topic = "水果与蔬菜的分类"
        
        statements = [
            KnowledgeStatement(
                id="fb_1",
                text="番茄在植物学上是水果，但在烹饪中常被当作蔬菜使用",
                source="Botanical Society",
                timestamp=datetime.now() - timedelta(days=100),
                quality_score=0.90,
                confidence=0.85,
                evidence_strength=0.80,
                is_noisy=False
            ),
            KnowledgeStatement(
                id="fb_2",
                text="黄瓜、南瓜、茄子在植物学上都是水果",
                source="Horticulture Journal",
                timestamp=datetime.now() - timedelta(days=80),
                quality_score=0.88,
                confidence=0.80,
                evidence_strength=0.75,
                is_noisy=False
            ),
            KnowledgeStatement(
                id="fb_3",
                text="在日常用语中，甜的植物果实被称为水果，咸的被称为蔬菜",
                source="Linguistics Study",
                timestamp=datetime.now() - timedelta(days=60),
                quality_score=0.75,
                confidence=0.70,
                evidence_strength=0.60,
                is_noisy=False
            ),
            KnowledgeStatement(
                id="fb_4",
                text="牛油果既是水果也是蔬菜，取决于使用场景",
                source="Culinary Institute",
                timestamp=datetime.now() - timedelta(days=40),
                quality_score=0.82,
                confidence=0.75,
                evidence_strength=0.70,
                is_noisy=False
            )
        ]
        
        return ComplexKnowledgeScenario(
            scenario_type=ScenarioType.FUZZY_BOUNDARIES,
            description="概念之间界限不清晰，需要上下文理解",
            topic=topic,
            statements=statements,
            expected_behavior="系统应理解上下文，处理模糊边界概念",
            success_criteria=[
                "上下文理解准确率 > 85%",
                "模糊概念正确处理",
                "多义性正确识别"
            ]
        )
    
    def generate_circular_dependency_scenario(self) -> ComplexKnowledgeScenario:
        """
        场景 6: 循环依赖知识
        
        特征：
        - 知识之间存在循环引用
        - 需要依赖图分析和循环检测
        """
        topic = "经济因素相互影响"
        
        statements = [
            KnowledgeStatement(
                id="cd_1",
                text="高就业率导致消费需求增加",
                source="Economics Journal",
                timestamp=datetime.now() - timedelta(days=90),
                quality_score=0.85,
                confidence=0.80,
                evidence_strength=0.75,
                is_noisy=False
            ),
            KnowledgeStatement(
                id="cd_2",
                text="消费需求增加推动企业扩张",
                source="Business Review",
                timestamp=datetime.now() - timedelta(days=85),
                quality_score=0.82,
                confidence=0.78,
                evidence_strength=0.72,
                is_noisy=False
            ),
            KnowledgeStatement(
                id="cd_3",
                text="企业扩张创造更多就业机会",
                source="Labor Statistics",
                timestamp=datetime.now() - timedelta(days=80),
                quality_score=0.80,
                confidence=0.75,
                evidence_strength=0.70,
                is_noisy=False
            ),
            KnowledgeStatement(
                id="cd_4",
                text="（形成循环：就业→消费→扩张→就业）",
                source="System Analysis",
                timestamp=datetime.now() - timedelta(days=70),
                quality_score=0.88,
                confidence=0.82,
                evidence_strength=0.78,
                is_noisy=False
            )
        ]
        
        return ComplexKnowledgeScenario(
            scenario_type=ScenarioType.CIRCULAR_DEPENDENCY,
            description="知识之间存在循环引用，需要循环检测",
            topic=topic,
            statements=statements,
            expected_behavior="系统应检测循环依赖，避免推理死循环",
            success_criteria=[
                "循环检测率 > 95%",
                "推理深度限制有效",
                "无死循环发生"
            ]
        )
    
    def generate_all_scenarios(self) -> List[ComplexKnowledgeScenario]:
        """生成所有六种复杂知识场景"""
        return [
            self.generate_high_noise_scenario(),
            self.generate_multi_source_conflict_scenario(),
            self.generate_knowledge_obsolescence_scenario(),
            self.generate_low_quality_injection_scenario(),
            self.generate_fuzzy_boundaries_scenario(),
            self.generate_circular_dependency_scenario()
        ]
    
    def print_scenario_summary(self, scenario: ComplexKnowledgeScenario):
        """打印场景摘要"""
        print(f"\n{'='*70}")
        print(f"场景: {scenario.scenario_type.value}")
        print(f"{'='*70}")
        print(f"主题: {scenario.topic}")
        print(f"描述: {scenario.description}")
        print(f"\n知识陈述数量: {len(scenario.statements)}")
        
        # 统计
        high_quality = sum(1 for s in scenario.statements if s.quality_score >= 0.7)
        noisy = sum(1 for s in scenario.statements if s.is_noisy)
        conflicting = sum(1 for s in scenario.statements if s.is_conflicting)
        outdated = sum(1 for s in scenario.statements if s.is_outdated)
        
        print(f"  高质量: {high_quality}")
        print(f"  噪声: {noisy}")
        print(f"  冲突: {conflicting}")
        print(f"  过时: {outdated}")
        
        print(f"\n预期行为: {scenario.expected_behavior}")
        print(f"\n成功标准:")
        for criterion in scenario.success_criteria:
            print(f"  - {criterion}")


def demo_complex_kb_generator():
    """演示复杂知识库生成器"""
    print("\n" + "📚 " * 35)
    print("Complex Knowledge Base Generator - 复杂知识库生成器")
    print("📚 " * 35)
    
    generator = ComplexKBGenerator()
    scenarios = generator.generate_all_scenarios()
    
    for scenario in scenarios:
        generator.print_scenario_summary(scenario)
        
        # 打印前3个陈述作为示例
        print(f"\n  示例陈述:")
        for stmt in scenario.statements[:3]:
            quality_bar = "█" * int(stmt.quality_score * 10)
            print(f"    [{stmt.id}] {stmt.text[:50]}...")
            print(f"      来源: {stmt.source}, 质量: {stmt.quality_score:.2f} {quality_bar}")
    
    print("\n" + "="*70)
    print(f"总计生成 {len(scenarios)} 个复杂知识场景")
    print("="*70)
    
    return scenarios


if __name__ == "__main__":
    demo_complex_kb_generator()
