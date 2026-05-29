"""
OpenBookQA 数据预处理
转换为 Stage 8 统一格式
"""

import json
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class Stage8Sample:
    """Stage 8 统一样本格式"""
    sample_id: str
    question: str
    context: str
    answer: str
    difficulty: str  # L1, L2, L3
    expected_gap: int  # 0=无缺口, 1=有缺口
    expected_retrieval: int  # 0=无需检索, 1=需要检索
    expected_policy: int  # 0=直接回答, 1=检索后回答
    knowledge_units: List[Dict]
    reasoning_chain: List[str]
    source: str


class OpenBookQAPreprocessor:
    """OpenBookQA 数据预处理器"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        
    def load_raw_data(self, split: str = "train") -> List[Dict]:
        """加载原始数据"""
        # 优先使用 main 版本
        main_file = self.data_dir / f"main_{split}.jsonl"
        additional_file = self.data_dir / f"additional_{split}.jsonl"
        
        data = []
        
        if main_file.exists():
            with open(main_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            print(f"[OpenBookQA] 加载 main {split}: {len(data)} 条")
        
        if additional_file.exists():
            with open(additional_file, 'r', encoding='utf-8') as f:
                additional_data = [json.loads(line.strip()) for line in f]
            print(f"[OpenBookQA] 加载 additional {split}: {len(additional_data)} 条")
            data.extend(additional_data)
        
        return data
    
    def preprocess(self, max_samples: int = 1000) -> List[Stage8Sample]:
        """
        预处理 OpenBookQA 数据
        
        OpenBookQA 特点:
        - 需要科学常识
        - 需要简单推理
        - 标记为 L2 级别
        """
        raw_data = self.load_raw_data("train")
        samples = []
        sample_id = 0
        
        for item in raw_data[:max_samples]:
            question = item.get('question_stem', item.get('question', ''))
            choices = item.get('choices', [])
            answer_key = item.get('answerKey', '')
            
            # 找到正确答案
            answer = ""
            for choice in choices:
                if choice.get('label') == answer_key:
                    answer = choice.get('text', '')
                    break
            
            # 构建 context (包含所有选项作为背景)
            context_parts = []
            for choice in choices:
                context_parts.append(f"{choice.get('label')}: {choice.get('text', '')}")
            context = " | ".join(context_parts)
            
            # OpenBookQA 需要常识+推理 -> L2
            difficulty = 'L2'
            expected_gap = 1  # 需要外部知识
            expected_retrieval = 1  # 需要检索
            expected_policy = 1  # 检索后回答
            
            # 提取知识单元
            knowledge_units = []
            for choice in choices:
                knowledge_units.append({
                    'type': 'concept',
                    'content': choice.get('text', ''),
                    'relation': 'option',
                })
            
            # 推理链
            reasoning_chain = [
                f"理解问题: {question}",
                "分析选项",
                f"选择答案: {answer_key}",
            ]
            
            sample = Stage8Sample(
                sample_id=f"obqa_{sample_id:05d}",
                question=question,
                context=context,
                answer=answer,
                difficulty=difficulty,
                expected_gap=expected_gap,
                expected_retrieval=expected_retrieval,
                expected_policy=expected_policy,
                knowledge_units=knowledge_units,
                reasoning_chain=reasoning_chain,
                source="openbookqa",
            )
            
            samples.append(sample)
            sample_id += 1
        
        print(f"[OpenBookQA] 预处理完成: {len(samples)} 条 L2 样本")
        return samples
    
    def save_to_jsonl(self, samples: List[Stage8Sample], output_path: str):
        """保存为 JSONL 格式"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps({
                    'sample_id': sample.sample_id,
                    'question': sample.question,
                    'context': sample.context,
                    'answer': sample.answer,
                    'difficulty': sample.difficulty,
                    'expected_gap': sample.expected_gap,
                    'expected_retrieval': sample.expected_retrieval,
                    'expected_policy': sample.expected_policy,
                    'knowledge_units': sample.knowledge_units,
                    'reasoning_chain': sample.reasoning_chain,
                    'source': sample.source,
                }, ensure_ascii=False) + '\n')
        
        print(f"[OpenBookQA] 保存到: {output_file}")


def main():
    """主函数"""
    print("="*70)
    print("OpenBookQA 数据预处理")
    print("="*70)
    
    # 1. 下载数据
    from download_openbookqa import download_and_save_openbookqa
    data_dir = download_and_save_openbookqa()
    
    # 2. 预处理
    preprocessor = OpenBookQAPreprocessor(data_dir)
    samples = preprocessor.preprocess(max_samples=1000)
    
    # 3. 保存
    output_path = "stage8_dataset/openbookqa_l2.jsonl"
    preprocessor.save_to_jsonl(samples, output_path)
    
    # 4. 统计
    print("\n" + "="*70)
    print("预处理统计")
    print("="*70)
    print(f"总样本数: {len(samples)}")
    print(f"难度分布: L2={len(samples)}")
    print(f"输出文件: {output_path}")
    print("="*70)


if __name__ == "__main__":
    main()
