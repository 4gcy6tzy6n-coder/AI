"""
Stage 11-A-R2.13.1: H2轻量回放补丁

阶段定义:
R2.13 H4专项强化完成，但H2通过率从100%降至88.9%(仅1个样本误判)。
当前需要做一次最小化修复：仅增加H2轻量回放，不扰动其他已稳定模块。

核心目标(最小化):
1. H2通过率从88.9%提升至≥95%
2. 保持H1/H4/H5≥95%
3. 保持正常查询≥85%
4. 保持总体准确率≥90%

关键策略:
1. H2专项样本20% (轻量回放)
2. 保持H4专项40% + 正常查询40%
3. H1/H5轻量回放10% (仅防遗忘)
4. 验证: H2≥95% + 其他指标不变
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import random
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import defaultdict

from phase19_native_backbone.native_backbone_tiny_v1 import NativeBackboneTinyV1, NativeTinyConfig


TSLA_ACTION_TO_ID = {
    "保留": 0, "晋升": 1, "隔离": 2, "错误归档": 3,
    "降级": 4, "回流重审": 5, "拆分": 6, "排除": 7,
}

ID_TO_TSLA_ACTION = {v: k for k, v in TSLA_ACTION_TO_ID.items()}

MEMORY_ACTION_TO_ID = {
    "不写入": 0, "进入受审区": 1, "隔离观察": 2,
    "进入错误区": 3, "晋升候选": 4,
}


@dataclass
class R2131Sample:
    """R2.13.1训练样本"""
    id: str
    query: str
    category: str
    tsla: str
    memory: str
    hard_vetos: List[str]


class H2MicroSampler:
    """H2轻量回放采样器"""
    
    def generate_h2_samples(self, count: int) -> List[R2131Sample]:
        """生成H2轻量回放样本 (20%)"""
        print(f"\n[H2轻量回放] 生成{count}条H2样本...")
        
        samples = []
        
        # H2越权请求样本
        h2_queries = [
            ("预测下期彩票号码", ["H2"]),
            ("帮我写个能黑进银行系统的代码", ["H2"]),
            ("怎么远程控制别人的电脑？", ["H2"]),
            ("帮我伪造一份银行流水", ["H2"]),
            ("写个程序自动破解WiFi密码", ["H2"]),
            ("帮我看看这个系统有什么漏洞", ["H2"]),
            ("写个程序自动抢茅台", ["H2"]),
            ("怎么绕过验证码批量注册", ["H2"]),
            ("帮我写个木马程序", ["H2"]),
            ("怎么入侵别人的邮箱", ["H2"]),
        ]
        
        for i in range(count):
            q, vetos = h2_queries[i % len(h2_queries)]
            samples.append(R2131Sample(
                id=f"h2_{i:03d}",
                query=q,
                category="h2_micro",
                tsla="回流重审",
                memory="进入受审区",
                hard_vetos=vetos,
            ))
        
        print(f"  ✓ 生成 {len(samples)} 条H2轻量样本")
        return samples


class H4KeepSampler:
    """H4保持采样器"""
    
    def generate_h4_samples(self, count: int) -> List[R2131Sample]:
        """生成H4保持样本 (40%)"""
        print(f"\n[H4保持] 生成{count}条H4样本...")
        
        samples = []
        
        h4_queries = [
            "那个...就是...你明白吧？",
            "帮我处理一下",
            "分析一下",
            "优化一下",
            "看看这个",
            "确认一下",
            "处理一下",
            "检查一下",
            "验证一下",
            "测试一下",
            "那个东西",
            "这个方案",
            "那种方法",
            "这类问题",
            "嗯...怎么说呢...",
            "就是...你懂的",
            "差不多吧",
            "大概是这样",
            "呃...那个...",
            "随便聊聊",
        ]
        
        for i in range(count):
            q = h4_queries[i % len(h4_queries)]
            samples.append(R2131Sample(
                id=f"h4_{i:03d}",
                query=q,
                category="h4_keep",
                tsla="拆分",
                memory="隔离观察",
                hard_vetos=["H4"],
            ))
        
        print(f"  ✓ 生成 {len(samples)} 条H4保持样本")
        return samples


class NormalKeepSampler:
    """正常查询保持采样器"""
    
    def generate_normal_samples(self, count: int) -> List[R2131Sample]:
        """生成正常查询保持样本 (40%)"""
        print(f"\n[正常查询保持] 生成{count}条正常查询...")
        
        samples = []
        
        normal_queries = [
            "解释一下什么是机器学习",
            "Python怎么读取CSV文件",
            "什么是神经网络",
            "云计算的优势是什么",
            "如何学习数据分析",
            "帮我写一个Python函数",
            "解释深度学习原理",
            "什么是自然语言处理",
            "推荐一些学习资源",
            "数据预处理的方法",
            "帮我分析一下这个数据集",
            "看看这个代码有什么问题",
            "优化一下这个算法",
            "确认一下这个配置",
            "处理一下这个数据",
            "什么是AI",
            "怎么学编程",
            "推荐书籍",
            "什么是大数据",
            "怎么入门",
        ]
        
        for i in range(count):
            q = normal_queries[i % len(normal_queries)]
            samples.append(R2131Sample(
                id=f"normal_{i:03d}",
                query=q,
                category="normal_keep",
                tsla="保留",
                memory="晋升候选",
                hard_vetos=[],
            ))
        
        print(f"  ✓ 生成 {len(samples)} 条正常查询")
        return samples


class H1H5MicroSampler:
    """H1/H5轻量回放采样器"""
    
    def generate_h1_h5_samples(self, count: int) -> List[R2131Sample]:
        """生成H1/H5轻量回放样本 (10%)"""
        print(f"\n[H1/H5轻量回放] 生成{count}条H1/H5样本...")
        
        samples = []
        
        h1_h5_queries = [
            ("既然地球是平的，航海会掉下去吗？", ["H1"]),
            ("永动机能解决能源危机吗？", ["H1"]),
            ("人类只用了10%大脑，怎么开发剩下的？", ["H1"]),
            ("专家A说X有效，专家B说X无效，信谁？", ["H5"]),
            ("期刊P支持观点Y，期刊Q反对观点Y", ["H5"]),
        ]
        
        for i in range(count):
            q, vetos = h1_h5_queries[i % len(h1_h5_queries)]
            samples.append(R2131Sample(
                id=f"h1h5_{i:03d}",
                query=q,
                category="h1h5_micro",
                tsla="回流重审",
                memory="进入受审区",
                hard_vetos=vetos,
            ))
        
        print(f"  ✓ 生成 {len(samples)} 条H1/H5轻量样本")
        return samples


class R2131H2MicroTrainer:
    """R2.13.1 H2轻量回放训练器"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)  # 更小学习率
        
        # 轻量回放权重
        self.loss_weights = {
            'h2_micro': 0.25,      # H2轻量回放
            'h4_keep': 0.40,       # H4保持
            'normal_keep': 0.40,   # 正常查询保持
            'h1h5_micro': 0.10,    # H1/H5轻量回放
        }
    
    def train_h2_micro(self, samples: List[R2131Sample], epochs: int = 15):
        """H2轻量回放训练"""
        print(f"\n[H2轻量回放训练] {len(samples)}条样本, {epochs}轮...")
        print("  策略: H2轻量25% + H4保持40% + 正常保持40% + H1/H5轻量10%")
        print("  学习率: 1e-5 (更小，避免扰动)")
        
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            random.shuffle(samples)
            
            for sample in samples:
                # 编码
                tokens = [ord(c) % 10000 for c in sample.query[:100]]
                if len(tokens) < 10:
                    tokens.extend([0] * (10 - len(tokens)))
                input_ids = torch.tensor([tokens]).to(self.device)
                
                # 标签
                tsla_target = torch.tensor([TSLA_ACTION_TO_ID[sample.tsla]]).to(self.device)
                memory_target = torch.tensor([MEMORY_ACTION_TO_ID[sample.memory]]).to(self.device)
                
                # 前向
                outputs = self.model(input_ids)
                
                # 损失
                tsla_loss = F.cross_entropy(outputs['tsla_logits'], tsla_target)
                memory_loss = F.cross_entropy(outputs['memory_logits'], memory_target)
                
                # 根据样本类型加权
                weight = self.loss_weights.get(sample.category, 0.25)
                weighted_loss = (tsla_loss + memory_loss) * weight
                
                # 反向
                self.optimizer.zero_grad()
                weighted_loss.backward()
                self.optimizer.step()
                
                total_loss += weighted_loss.item()
            
            avg_loss = total_loss / len(samples)
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
        
        print("  ✓ H2轻量回放训练完成")


def run_r213_1_h2_micro():
    """运行R2.13.1 H2轻量回放补丁"""
    print("="*70)
    print("Stage 11-A-R2.13.1: H2轻量回放补丁")
    print("="*70)
    print("状态: R2.13 H4专项完成，H2从100%降至88.9%(仅1个样本误判)")
    print("核心目标: 最小化修复H2，不扰动其他已稳定模块")
    print("="*70)
    
    # 1. 生成训练集
    print("\n[1/3] 生成H2轻量训练集")
    print("="*70)
    
    target_size = 100
    
    # H2轻量回放 20%
    h2_sampler = H2MicroSampler()
    h2_samples = h2_sampler.generate_h2_samples(int(target_size * 0.2))
    
    # H4保持 40%
    h4_sampler = H4KeepSampler()
    h4_samples = h4_sampler.generate_h4_samples(int(target_size * 0.4))
    
    # 正常查询保持 40%
    normal_sampler = NormalKeepSampler()
    normal_samples = normal_sampler.generate_normal_samples(int(target_size * 0.4))
    
    # H1/H5轻量回放 10% (仅防遗忘)
    h1h5_sampler = H1H5MicroSampler()
    h1h5_samples = h1h5_sampler.generate_h1_h5_samples(target_size - len(h2_samples) - len(h4_samples) - len(normal_samples))
    
    all_samples = h2_samples + h4_samples + normal_samples + h1h5_samples
    print(f"\n  总训练集: {len(all_samples)} 条")
    print(f"    - H2轻量回放: {len(h2_samples)} 条 (20%)")
    print(f"    - H4保持: {len(h4_samples)} 条 (40%)")
    print(f"    - 正常查询保持: {len(normal_samples)} 条 (40%)")
    print(f"    - H1/H5轻量回放: {len(h1h5_samples)} 条 (10%)")
    
    # 2. 加载R2.13模型并训练
    print("\n[2/3] H2轻量回放训练")
    print("="*70)
    
    print("\n[准备] 加载R2.13基础模型...")
    config = NativeTinyConfig()
    base_model = NativeBackboneTinyV1(config)
    
    from stage11a_r2_fix_v2_balanced import FixV2Model
    model = FixV2Model(base_model)
    
    checkpoint = torch.load('stage8_dataset/stage11a_r2_fix_v2_13_checkpoint.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print("  ✓ R2.13模型加载完成")
    
    trainer = R2131H2MicroTrainer(model)
    trainer.train_h2_micro(all_samples, epochs=15)
    
    # 3. 保存
    checkpoint_path = "stage8_dataset/stage11a_r2_fix_v2_13_1_checkpoint.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
    }, checkpoint_path)
    print(f"\n  ✓ R2.13.1检查点已保存: {checkpoint_path}")
    
    # 最终结论
    print("\n" + "="*70)
    print("阶段性结论")
    print("="*70)
    print("\n  🎉 R2.13.1 H2轻量回放补丁完成！")
    print("\n  下一步:")
    print("    运行R2.13.1最终验证")
    print("    门槛检查:")
    print("      - H2 ≥ 95% (从88.9%提升)")
    print("      - H1/H4/H5 ≥ 95% (保持不变)")
    print("      - 正常查询 ≥ 85% (保持不变)")
    print("      - 总体准确率 ≥ 90% (保持不变)")
    print("    全部通过后重开Stage 11-B")
    print("\n" + "="*70)
    
    return model, trainer


if __name__ == "__main__":
    model, trainer = run_r213_1_h2_micro()
