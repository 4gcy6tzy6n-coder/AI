"""
Stage 8 数据预处理脚本

将原始数据集 (SQuAD 2.0, OpenBookQA) 转换为 Stage 8 统一格式
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class Stage8Sample:
    """Stage 8 标准样本格式"""
    id: str
    source: str
    question: str
    context: Optional[str]
    difficulty: str  # L1/L2/L3
    domain: str
    expected_gap: int  # 0/1
    expected_retrieval: int  # 0/1
    expected_policy: int  # 0-3
    answer: str
    explanation: str
    knowledge_units: List[str]
    reasoning_chain: List[str]
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'source': self.source,
            'question': self.question,
            'context': self.context,
            'difficulty': self.difficulty,
            'domain': self.domain,
            'expected_gap': self.expected_gap,
            'expected_retrieval': self.expected_retrieval,
            'expected_policy': self.expected_policy,
            'answer': self.answer,
            'explanation': self.explanation,
            'knowledge_units': self.knowledge_units,
            'reasoning_chain': self.reasoning_chain,
        }


class SQuADPreprocessor:
    """SQuAD 2.0 数据预处理器"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        
    def load_raw_data(self) -> List[Dict]:
        """加载原始数据"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data['data']
    
    def preprocess(self, max_samples: int = 300) -> List[Stage8Sample]:
        """
        预处理 SQuAD 数据
        
        策略:
        - 有答案的问题 -> L1 (直接检索)
        - 无答案的问题 (is_impossible=True) -> 需要推理，标记为 L2
        """
        raw_data = self.load_raw_data()
        samples = []
        sample_id = 0
        
        for article in raw_data:
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                
                for qa in paragraph['qas']:
                    if sample_id >= max_samples:
                        break
                    
                    question = qa['question']
                    is_impossible = qa.get('is_impossible', False)
                    
                    # 判断难度
                    if is_impossible:
                        difficulty = 'L2'
                        expected_gap = 1  # 需要新知识
                        expected_retrieval = 1
                        expected_policy = 1  # 检索策略
                        answer = "无法从上下文中找到答案"
                    else:
                        difficulty = 'L1'
                        expected_gap = 0  # 知识在上下文中
                        expected_retrieval = 1
                        expected_policy = 0  # 直接回答
                        
                        # 取第一个答案
                        if qa['answers']:
                            answer = qa['answers'][0]['text']
                        else:
                            continue
                    
                    # 提取知识单元 (简化版：从上下文提取关键词)
                    knowledge_units = self._extract_knowledge_units(context, question)
                    
                    sample = Stage8Sample(
                        id=f'squad_{sample_id:04d}',
                        source='SQuAD 2.0',
                        question=question,
                        context=context[:500] if len(context) > 500 else context,  # 截断长文本
                        difficulty=difficulty,
                        domain='general',
                        expected_gap=expected_gap,
                        expected_retrieval=expected_retrieval,
                        expected_policy=expected_policy,
                        answer=answer,
                        explanation=f'基于上下文: {context[:200]}...',
                        knowledge_units=knowledge_units,
                        reasoning_chain=[],
                    )
                    
                    samples.append(sample)
                    sample_id += 1
                
                if sample_id >= max_samples:
                    break
            if sample_id >= max_samples:
                break
        
        return samples
    
    def _extract_knowledge_units(self, context: str, question: str) -> List[str]:
        """提取知识单元 (简化实现)"""
        # 这里可以使用更复杂的 NLP 方法
        # 简化版：提取问题中的名词作为知识单元
        words = question.lower().split()
        stop_words = {'what', 'is', 'are', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'of'}
        units = [w for w in words if w not in stop_words and len(w) > 3]
        return units[:3]  # 最多3个


class OpenBookQAPreprocessor:
    """OpenBookQA 数据预处理器"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        
    def load_raw_data(self) -> List[Dict]:
        """加载原始数据"""
        samples = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                samples.append(json.loads(line.strip()))
        return samples
    
    def preprocess(self, max_samples: int = 200) -> List[Stage8Sample]:
        """
        预处理 OpenBookQA 数据
        
        策略:
        - 所有问题标记为 L2 (需要结合知识推理)
        """
        raw_data = self.load_raw_data()
        samples = []
        
        for i, item in enumerate(raw_data[:max_samples]):
            question = item['question']['stem']
            choices = item['question']['choices']
            answer_key = item['answerKey']
            
            # 找到正确答案
            answer = next((c['text'] for c in choices if c['label'] == answer_key), "")
            
            # 构建解释 (结合选项)
            choice_text = "; ".join([f"{c['label']}: {c['text']}" for c in choices])
            explanation = f"选项: {choice_text}. 正确答案: {answer}"
            
            # 提取知识单元
            knowledge_units = self._extract_knowledge_units(question)
            
            sample = Stage8Sample(
                id=f'obqa_{i:04d}',
                source='OpenBookQA',
                question=question,
                context=None,  # OpenBookQA 依赖外部知识
                difficulty='L2',
                domain='science',
                expected_gap=1,  # 需要外部知识
                expected_retrieval=1,
                expected_policy=2,  # 推理策略
                answer=answer,
                explanation=explanation,
                knowledge_units=knowledge_units,
                reasoning_chain=['分析选项', '检索知识', '推理判断'],
            )
            
            samples.append(sample)
        
        return samples
    
    def _extract_knowledge_units(self, question: str) -> List[str]:
        """提取知识单元"""
        words = question.lower().split()
        stop_words = {'what', 'is', 'are', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'of', 'if'}
        units = [w for w in words if w not in stop_words and len(w) > 3]
        return units[:3]


class Stage8DatasetBuilder:
    """Stage 8 数据集构建器"""
    
    def __init__(self, raw_data_dir: str, output_dir: str):
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def build_mvp_dataset(self) -> Dict[str, List[Stage8Sample]]:
        """
        构建 MVP 数据集 (100 条)
        
        分布:
        - SQuAD L1: 70 条
        - OpenBookQA L2: 30 条
        """
        print("="*70)
        print("构建 Stage 8 MVP 数据集")
        print("="*70)
        
        # 1. 处理 SQuAD
        print("\n[1/2] 处理 SQuAD 2.0...")
        squad_path = self.raw_data_dir / 'train-v2.0.json'
        if squad_path.exists():
            squad_preprocessor = SQuADPreprocessor(str(squad_path))
            squad_samples = squad_preprocessor.preprocess(max_samples=70)
            print(f"  ✓ 生成 {len(squad_samples)} 条 L1 样本")
        else:
            print(f"  ✗ 未找到 SQuAD 数据: {squad_path}")
            squad_samples = []
        
        # 2. 处理 OpenBookQA
        print("\n[2/2] 处理 OpenBookQA...")
        obqa_path = self.raw_data_dir / 'OpenBookQA' / 'Main' / 'train.jsonl'
        if not obqa_path.exists():
            # 尝试其他路径
            obqa_path = self.raw_data_dir / 'train.jsonl'
        
        if obqa_path.exists():
            obqa_preprocessor = OpenBookQAPreprocessor(str(obqa_path))
            obqa_samples = obqa_preprocessor.preprocess(max_samples=30)
            print(f"  ✓ 生成 {len(obqa_samples)} 条 L2 样本")
        else:
            print(f"  ✗ 未找到 OpenBookQA 数据: {obqa_path}")
            obqa_samples = []
        
        # 3. 合并并划分
        all_samples = squad_samples + obqa_samples
        random.seed(42)
        random.shuffle(all_samples)
        
        # MVP: 80 训练 + 20 验证
        train_samples = all_samples[:80]
        val_samples = all_samples[80:100]
        
        print(f"\n[数据集统计]")
        print(f"  总样本数: {len(all_samples)}")
        print(f"  训练集: {len(train_samples)}")
        print(f"  验证集: {len(val_samples)}")
        
        return {
            'train': train_samples,
            'val': val_samples,
        }
    
    def save_dataset(self, dataset: Dict[str, List[Stage8Sample]]):
        """保存数据集"""
        print("\n[保存数据集]")
        
        for split, samples in dataset.items():
            output_file = self.output_dir / f'{split}.jsonl'
            with open(output_file, 'w', encoding='utf-8') as f:
                for sample in samples:
                    f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + '\n')
            print(f"  ✓ {split}: {len(samples)} 条 -> {output_file}")
        
        # 保存统计信息
        stats = {
            'total_samples': sum(len(s) for s in dataset.values()),
            'splits': {k: len(v) for k, v in dataset.items()},
            'difficulty_distribution': self._count_difficulty(dataset),
        }
        
        stats_file = self.output_dir / 'stats.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"  ✓ 统计信息 -> {stats_file}")
    
    def _count_difficulty(self, dataset: Dict[str, List[Stage8Sample]]) -> Dict:
        """统计难度分布"""
        distribution = {}
        for split, samples in dataset.items():
            distribution[split] = {}
            for sample in samples:
                diff = sample.difficulty
                distribution[split][diff] = distribution[split].get(diff, 0) + 1
        return distribution


def main():
    """主函数"""
    # 数据路径
    raw_data_dir = r'E:\new ai\data'  # 原始数据目录
    output_dir = r'd:\post_transformer_ai\phase20_self_construct\stage8_dataset'
    
    print(f"原始数据目录: {raw_data_dir}")
    print(f"输出目录: {output_dir}")
    
    # 检查原始数据
    raw_path = Path(raw_data_dir)
    print(f"\n[检查原始数据]")
    print(f"  目录存在: {raw_path.exists()}")
    if raw_path.exists():
        files = list(raw_path.iterdir())
        print(f"  文件数量: {len(files)}")
        for f in files[:10]:  # 显示前10个
            print(f"    - {f.name}")
    
    # 构建数据集
    builder = Stage8DatasetBuilder(raw_data_dir, output_dir)
    dataset = builder.build_mvp_dataset()
    
    # 保存
    if dataset['train'] or dataset['val']:
        builder.save_dataset(dataset)
        print("\n" + "="*70)
        print("✓ MVP 数据集构建完成!")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("✗ 未找到有效数据，请检查数据路径")
        print("="*70)


if __name__ == "__main__":
    main()
