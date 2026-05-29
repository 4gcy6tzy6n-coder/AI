"""
下载 OpenBookQA 数据集
使用 HuggingFace datasets 库
"""

from datasets import load_dataset
import json
from pathlib import Path


def download_and_save_openbookqa(save_dir: str = "E:/new ai/data/openbookqa"):
    """
    下载 OpenBookQA 数据集并保存为 JSON 格式
    
    Args:
        save_dir: 保存目录
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("下载 OpenBookQA 数据集")
    print("="*70)
    
    # 下载 main 版本
    print("\n[1/2] 下载 main 版本...")
    ds_main = load_dataset("allenai/openbookqa", "main")
    
    for split in ['train', 'validation', 'test']:
        if split in ds_main:
            output_file = save_path / f"main_{split}.jsonl"
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in ds_main[split]:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"  ✓ {split}: {len(ds_main[split])} 条 -> {output_file}")
    
    # 下载 additional 版本 (包含额外数据)
    print("\n[2/2] 下载 additional 版本...")
    ds_additional = load_dataset("allenai/openbookqa", "additional")
    
    for split in ['train', 'validation', 'test']:
        if split in ds_additional:
            output_file = save_path / f"additional_{split}.jsonl"
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in ds_additional[split]:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"  ✓ {split}: {len(ds_additional[split])} 条 -> {output_file}")
    
    print("\n" + "="*70)
    print("下载完成！")
    print(f"数据保存位置: {save_path}")
    print("="*70)
    
    return save_path


if __name__ == "__main__":
    download_and_save_openbookqa()
