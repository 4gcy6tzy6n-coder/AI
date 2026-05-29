"""
手动下载 OpenBookQA 数据集
从官方 GitHub 仓库下载
"""

import urllib.request
import json
import os
from pathlib import Path
import ssl

# 禁用 SSL 验证 (用于解决证书问题)
ssl._create_default_https_context = ssl._create_unverified_context


def download_file(url: str, output_path: str):
    """下载文件"""
    try:
        urllib.request.urlretrieve(url, output_path)
        return True
    except Exception as e:
        print(f"  ✗ 下载失败: {e}")
        return False


def download_openbookqa_manual(save_dir: str = "E:/new ai/data/openbookqa"):
    """
    手动下载 OpenBookQA 数据集
    
    从官方 GitHub 仓库下载
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("手动下载 OpenBookQA 数据集")
    print("="*70)
    
    # OpenBookQA GitHub 原始文件 URL (Public 分支)
    base_url = "https://raw.githubusercontent.com/allenai/OpenBookQA/public/data/main"
    
    files = {
        "train": f"{base_url}/train.jsonl",
        "dev": f"{base_url}/dev.jsonl",
        "test": f"{base_url}/test.jsonl",
    }
    
    downloaded = []
    
    for split, url in files.items():
        output_file = save_path / f"{split}.jsonl"
        print(f"\n[{split}]")
        print(f"  URL: {url}")
        print(f"  保存到: {output_file}")
        
        if download_file(url, str(output_file)):
            # 统计行数
            with open(output_file, 'r', encoding='utf-8') as f:
                count = sum(1 for _ in f)
            print(f"  ✓ 下载成功: {count} 条")
            downloaded.append((split, count))
        else:
            print(f"  ✗ 下载失败")
    
    print("\n" + "="*70)
    print("下载完成！")
    print("="*70)
    for split, count in downloaded:
        print(f"  {split}: {count} 条")
    print(f"\n数据保存位置: {save_path}")
    print("="*70)
    
    return save_path, downloaded


if __name__ == "__main__":
    download_openbookqa_manual()
