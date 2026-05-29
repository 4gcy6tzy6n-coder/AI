"""
候选到微调任务转换器

将高质量候选编译成有监督微调样本

候选: relation_dc371e5e27f4
类型: RELATION (检索 → 知识库)
内容: {"relation_type": "uses", "entities": {"a": "检索", "b": "知识库"}}
"""

import json
import random
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class FinetuneSample:
    """微调样本"""
    input_text: str
    target_gap: int      # 0=NO_GAP, 1=RETRIEVABLE, 2=HIGH_RISK
    target_strategy: int # 0=DIRECT, 1=RETRIEVAL_FIRST, 2=CONSERVATIVE, 3=DECLINE, 4=REVIEW
    sample_type: str     # positive, negative, adversarial


class CandidateToFinetuneTask:
    """候选到微调任务转换器"""
    
    def __init__(self, candidate: Dict):
        self.candidate = candidate
        self.relation_type = candidate.get('generated_content', {}).get('relation_type', 'uses')
        self.entities = candidate.get('entities', {'a': '检索', 'b': '知识库'})
        self.entity_a = self.entities.get('a', 'A')
        self.entity_b = self.entities.get('b', 'B')
    
    def generate_positive_samples(self, num: int = 15) -> List[FinetuneSample]:
        """
        生成正样本
        
        明确应该识别出该关系的例子
        """
        templates = [
            # 直接询问关系
            "{a} 和 {b} 是什么关系？",
            "{a} 与 {b} 之间有什么联系？",
            "请解释 {a} 和 {b} 的关系",
            "{a} 如何利用 {b}？",
            "{b} 在 {a} 中起什么作用？",
            
            # 场景化询问
            "在使用 {a} 时，为什么需要 {b}？",
            "{a} 依赖 {b} 吗？",
            "{a} 和 {b} 是如何配合工作的？",
            "没有 {b}，{a} 还能工作吗？",
            "{a} 为什么离不开 {b}？",
            
            # 技术角度
            "从技术角度看，{a} 和 {b} 的关系是什么？",
            "{a} 的实现对 {b} 有什么要求？",
            "{b} 如何支撑 {a} 的功能？",
        ]
        
        samples = []
        for i in range(num):
            template = templates[i % len(templates)]
            text = template.format(a=self.entity_a, b=self.entity_b)
            
            samples.append(FinetuneSample(
                input_text=text,
                target_gap=1,  # RETRIEVABLE - 需要检索来确认关系
                target_strategy=1,  # RETRIEVAL_FIRST - 先检索再回答
                sample_type='positive'
            ))
        
        return samples
    
    def generate_negative_samples(self, num: int = 15) -> List[FinetuneSample]:
        """
        生成负样本
        
        表面相似但不该触发该关系的例子
        """
        # 无关实体对
        unrelated_pairs = [
            ("苹果", "香蕉"),
            ("汽车", "飞机"),
            ("猫", "狗"),
            ("桌子", "椅子"),
            ("手机", "电脑"),
        ]
        
        templates = [
            "{a} 和 {b} 是什么关系？",
            "{a} 与 {b} 之间有什么联系？",
            "请解释 {a} 和 {b} 的关系",
        ]
        
        samples = []
        for i in range(num):
            pair = unrelated_pairs[i % len(unrelated_pairs)]
            template = templates[i % len(templates)]
            text = template.format(a=pair[0], b=pair[1])
            
            samples.append(FinetuneSample(
                input_text=text,
                target_gap=0,  # NO_GAP - 无关实体，不需要检索
                target_strategy=0,  # DIRECT - 直接回答
                sample_type='negative'
            ))
        
        return samples
    
    def generate_adversarial_samples(self, num: int = 20) -> List[FinetuneSample]:
        """
        生成对抗样本
        
        同义改写、口语化改写、边界模糊样本
        """
        # 同义改写
        synonym_templates = [
            "{a} 和 {b} 的关联是什么？",
            "{a} 与 {b} 有何关联？",
            "{a} 对 {b} 的依赖程度如何？",
            "{a} 与 {b} 的耦合关系是怎样的？",
            "{a} 和 {b} 是如何相互作用的？",
        ]
        
        # 口语化改写
        colloquial_templates = [
            "{a} 跟 {b} 有啥关系啊？",
            "为啥 {a} 要用 {b} 呢？",
            "{a} 是不是离不开 {b}？",
            "{a} 和 {b} 是绑在一起的吗？",
            "用 {a} 的时候为啥还得有 {b}？",
        ]
        
        # 边界模糊样本（相关但不完全相同的关系）
        boundary_templates = [
            "{a} 和 {b} 的区别是什么？",  # 询问区别而非关系
            "{a} 比 {b} 好吗？",  # 比较而非关系
            "除了 {b}，{a} 还能用什么？",  # 替代方案
            "{a} 和 {b} 哪个更重要？",  # 优先级
        ]
        
        samples = []
        
        # 同义改写 (8条)
        for i in range(8):
            template = synonym_templates[i % len(synonym_templates)]
            text = template.format(a=self.entity_a, b=self.entity_b)
            samples.append(FinetuneSample(
                input_text=text,
                target_gap=1,
                target_strategy=1,
                sample_type='adversarial_synonym'
            ))
        
        # 口语化改写 (8条)
        for i in range(8):
            template = colloquial_templates[i % len(colloquial_templates)]
            text = template.format(a=self.entity_a, b=self.entity_b)
            samples.append(FinetuneSample(
                input_text=text,
                target_gap=1,
                target_strategy=1,
                sample_type='adversarial_colloquial'
            ))
        
        # 边界模糊 (4条)
        for i in range(4):
            template = boundary_templates[i % len(boundary_templates)]
            text = template.format(a=self.entity_a, b=self.entity_b)
            samples.append(FinetuneSample(
                input_text=text,
                target_gap=1,  # 仍需要检索
                target_strategy=2,  # CONSERVATIVE - 保守回答
                sample_type='adversarial_boundary'
            ))
        
        return samples
    
    def generate_all_samples(self) -> List[FinetuneSample]:
        """生成所有样本"""
        positive = self.generate_positive_samples(15)
        negative = self.generate_negative_samples(15)
        adversarial = self.generate_adversarial_samples(20)
        
        all_samples = positive + negative + adversarial
        
        # 打乱顺序
        random.shuffle(all_samples)
        
        return all_samples
    
    def save_to_jsonl(self, output_path: str = "finetune_data/relation_finetune_samples.jsonl"):
        """保存为 JSONL 格式"""
        samples = self.generate_all_samples()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                data = {
                    'input_text': sample.input_text,
                    'target_gap': sample.target_gap,
                    'target_strategy': sample.target_strategy,
                    'sample_type': sample.sample_type,
                }
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
        
        print(f"✓ 生成 {len(samples)} 条微调样本")
        print(f"  - 正样本: 15")
        print(f"  - 负样本: 15")
        print(f"  - 对抗样本: 20")
        print(f"  保存到: {output_path}")
        
        return samples


def load_candidate(candidate_id: str = "relation_dc371e5e27f4") -> Dict:
    """加载候选"""
    # 从原始候选文件加载完整信息
    path = "candidates/stage5b_high_quality_batch.jsonl"
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if data.get('candidate_id') == candidate_id:
                return data
    
    return None


def main():
    """主函数"""
    print("=" * 70)
    print("候选到微调任务转换")
    print("=" * 70)
    
    # 加载候选
    candidate = load_candidate()
    if not candidate:
        print("✗ 未找到候选")
        return
    
    print(f"\n候选信息:")
    print(f"  ID: {candidate['candidate_id']}")
    print(f"  类型: RELATION")
    print(f"  实体: {candidate['entities']}")
    print(f"  关系: {candidate['generated_content']['relation_type']}")
    
    # 创建转换器
    converter = CandidateToFinetuneTask(candidate)
    
    # 生成样本
    samples = converter.save_to_jsonl()
    
    # 显示样本示例
    print("\n样本示例:")
    for i, sample in enumerate(samples[:5]):
        print(f"\n  [{i+1}] {sample.sample_type}")
        print(f"      输入: {sample.input_text}")
        print(f"      目标: gap={sample.target_gap}, strategy={sample.target_strategy}")


if __name__ == "__main__":
    main()
