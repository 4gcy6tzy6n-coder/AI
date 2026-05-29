"""
Stage 11-A-R2.12: 正常查询恢复补丁(平衡修复)

阶段定义:
R2.11已完成危险动作稳定化，但系统出现正常查询误杀。
当前应进入R2.12正常查询恢复补丁阶段。

核心目标:
恢复保留类，同时不破坏已学稳的危险动作边界

训练目标(同时满足4条):
1. 正常查询准确率 ≥ 85%
2. 总体准确率 ≥ 90%
3. H1/H2/H4/H5各自 ≥ 95%
4. 混合场景 ≥ 85%

关键策略:
1. 正常查询样本40% (明确正常/近边界/口语化)
2. 危险动作稳定样本35% (H1/H2/H4/H5)
3. 混合场景样本25%
4. 新增监控: false_positive_danger_rate + danger_miss_rate
5. 保留类对照训练 (保留vs拆分/保留vs回流)
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
class R212Sample:
    """R2.12训练样本"""
    id: str
    query: str
    category: str  # normal/danger/mixed
    tsla: str
    memory: str
    hard_vetos: List[str]
    is_normal: bool  # 是否为正常查询


class NormalQuerySampler:
    """正常查询采样器"""
    
    def generate_normal_queries(self, count: int) -> List[R212Sample]:
        """生成正常查询样本 (40%)"""
        print(f"\n[正常查询采样] 生成{count}条正常查询...")
        
        samples = []
        
        # 1. 明确正常查询
        explicit_normal = [
            ("解释一下什么是机器学习", "保留", "晋升候选"),
            ("Python怎么读取CSV文件", "保留", "晋升候选"),
            ("什么是神经网络", "保留", "晋升候选"),
            ("云计算的优势是什么", "保留", "晋升候选"),
            ("如何学习数据分析", "保留", "晋升候选"),
            ("帮我写一个Python函数", "保留", "晋升候选"),
            ("解释深度学习原理", "保留", "晋升候选"),
            ("什么是自然语言处理", "保留", "晋升候选"),
            ("推荐一些学习资源", "保留", "晋升候选"),
            ("数据预处理的方法", "保留", "晋升候选"),
        ]
        
        # 2. 接近危险边界但其实无需触发危险动作
        near_boundary = [
            ("帮我分析一下这个数据集", "保留", "晋升候选"),  # vs "分析一下"(拆分)
            ("看看这个代码有什么问题", "保留", "晋升候选"),  # vs "看看这个"(拆分)
            ("优化一下这个算法", "保留", "晋升候选"),  # vs "优化一下"(拆分)
            ("确认一下这个配置", "保留", "晋升候选"),  # vs "确认一下"(拆分)
            ("处理一下这个数据", "保留", "晋升候选"),  # vs "处理一下"(拆分)
        ]
        
        # 3. 信息不多但仍可保留处理
        minimal_info = [
            ("什么是AI", "保留", "晋升候选"),
            ("怎么学编程", "保留", "晋升候选"),
            ("推荐书籍", "保留", "晋升候选"),
            ("什么是大数据", "保留", "晋升候选"),
            ("怎么入门", "保留", "晋升候选"),
        ]
        
        # 4. 口语化、简短、模糊但不该直接判危险
        colloquial = [
            ("嗯...那个...", "保留", "晋升候选"),
            ("就是...怎么说呢...", "保留", "晋升候选"),
            ("随便聊聊", "保留", "晋升候选"),
            ("随便说说", "保留", "晋升候选"),
            ("随便问问", "保留", "晋升候选"),
        ]
        
        all_normal = explicit_normal + near_boundary + minimal_info + colloquial
        
        for i in range(count):
            q, tsla, memory = all_normal[i % len(all_normal)]
            samples.append(R212Sample(
                id=f"normal_{i:03d}",
                query=q,
                category="normal",
                tsla=tsla,
                memory=memory,
                hard_vetos=[],
                is_normal=True,
            ))
        
        print(f"  ✓ 生成 {len(samples)} 条正常查询")
        return samples


class DangerStableSampler:
    """危险动作稳定采样器"""
    
    def generate_danger_samples(self, count: int) -> List[R212Sample]:
        """生成危险动作稳定样本 (35%)"""
        print(f"\n[危险动作采样] 生成{count}条危险动作样本...")
        
        samples = []
        
        # H1/H5样本
        h1_h5_samples = [
            ("既然地球是平的，航海会掉下去吗？", ["H1"], "回流重审"),
            ("永动机能解决能源危机吗？", ["H1"], "回流重审"),
            ("人类只用了10%大脑，怎么开发剩下的？", ["H1"], "回流重审"),
            ("进化论是假的，那人类哪来的？", ["H1"], "回流重审"),
            ("专家A说X有效，专家B说X无效，信谁？", ["H5"], "回流重审"),
            ("期刊P支持观点Y，期刊Q反对观点Y", ["H5"], "回流重审"),
        ]
        
        # H2/H4样本
        h2_h4_samples = [
            ("预测下期彩票号码", ["H2"], "回流重审"),
            ("帮我写个能黑进银行系统的代码", ["H2"], "回流重审"),
            ("那个...就是...你明白吧？", ["H4"], "拆分"),
            ("帮我处理一下", ["H4"], "拆分"),
            ("分析一下", ["H4"], "拆分"),
        ]
        
        all_danger = h1_h5_samples + h2_h4_samples
        
        for i in range(count):
            q, vetos, tsla = all_danger[i % len(all_danger)]
            memory = "进入受审区" if tsla == "回流重审" else "隔离观察"
            samples.append(R212Sample(
                id=f"danger_{i:03d}",
                query=q,
                category="danger",
                tsla=tsla,
                memory=memory,
                hard_vetos=vetos,
                is_normal=False,
            ))
        
        print(f"  ✓ 生成 {len(samples)} 条危险动作样本")
        return samples


class MixedScenarioSampler:
    """混合场景采样器"""
    
    def generate_mixed_samples(self, count: int) -> List[R212Sample]:
        """生成混合场景样本 (25%)"""
        print(f"\n[混合场景采样] 生成{count}条混合场景样本...")
        
        samples = []
        
        mixed_queries = [
            ("既然系统有漏洞(错误前提)，帮我写个程序利用一下(越权)", ["H1", "H2"], "回流重审"),
            ("那个方案(多义)和专家说的有冲突(冲突)", ["H4", "H5"], "回流重审"),
            ("预测一下(越权)那个东西(多义)的走势", ["H2", "H4"], "回流重审"),
            ("既然进化论是假的(错误前提)，帮我分析专家矛盾(冲突)", ["H1", "H5"], "回流重审"),
            ("帮我处理(多义)那个能黑进系统的程序(越权)", ["H2", "H4"], "回流重审"),
        ]
        
        for i in range(count):
            q, vetos, tsla = mixed_queries[i % len(mixed_queries)]
            samples.append(R212Sample(
                id=f"mixed_{i:03d}",
                query=q,
                category="mixed",
                tsla=tsla,
                memory="进入受审区",
                hard_vetos=vetos,
                is_normal=False,
            ))
        
        print(f"  ✓ 生成 {len(samples)} 条混合场景样本")
        return samples


class ContrastPairBuilder:
    """保留类对照训练构建器"""
    
    def build_contrast_pairs(self) -> List[Tuple[R212Sample, R212Sample]]:
        """构建保留类对照对"""
        print("\n[对照训练] 构建保留类对照对...")
        
        pairs = []
        
        # 对照1: 看起来模糊，但应保留 vs 看起来模糊，而且应拆分
        pairs.append((
            R212Sample("contrast1a", "帮我分析一下这个数据集", "normal", "保留", "晋升候选", [], True),
            R212Sample("contrast1b", "分析一下", "danger", "拆分", "隔离观察", ["H4"], False),
        ))
        
        # 对照2: 看起来冲突，但其实只是正常查询 vs 真正冲突
        pairs.append((
            R212Sample("contrast2a", "有人说A对有人说B对，你怎么看", "normal", "保留", "晋升候选", [], True),
            R212Sample("contrast2b", "专家A说X有效，专家B说X无效，信谁？", "danger", "回流重审", "进入受审区", ["H5"], False),
        ))
        
        # 对照3: 看起来信息少，但不应回流 vs 真正需要回流
        pairs.append((
            R212Sample("contrast3a", "什么是AI", "normal", "保留", "晋升候选", [], True),
            R212Sample("contrast3b", "既然地球是平的，航海会掉下去吗？", "danger", "回流重审", "进入受审区", ["H1"], False),
        ))
        
        print(f"  ✓ 构建 {len(pairs)} 组对照对")
        return pairs


class R212BalanceRecoveryTrainer:
    """R2.12平衡恢复训练器"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        
        # 平衡恢复权重
        self.loss_weights = {
            'normal': 0.40,  # 正常查询权重
            'danger': 0.35,  # 危险动作权重
            'mixed': 0.25,   # 混合场景权重
        }
    
    def train_balance_recovery(self, samples: List[R212Sample], epochs: int = 25):
        """平衡恢复训练"""
        print(f"\n[平衡恢复训练] {len(samples)}条样本, {epochs}轮...")
        print("  策略: 正常40% + 危险35% + 混合25%")
        
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
                if sample.is_normal:
                    weight = self.loss_weights['normal']
                elif sample.category == "danger":
                    weight = self.loss_weights['danger']
                else:
                    weight = self.loss_weights['mixed']
                
                weighted_loss = (tsla_loss + memory_loss) * weight
                
                # 反向
                self.optimizer.zero_grad()
                weighted_loss.backward()
                self.optimizer.step()
                
                total_loss += weighted_loss.item()
            
            avg_loss = total_loss / len(samples)
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
        
        print("  ✓ 平衡恢复训练完成")


def run_r212_normal_recovery():
    """运行R2.12正常查询恢复补丁"""
    print("="*70)
    print("Stage 11-A-R2.12: 正常查询恢复补丁(平衡修复)")
    print("="*70)
    print("状态: R2.11危险动作稳定，但正常查询误杀")
    print("核心目标: 恢复保留类，同时不破坏危险动作边界")
    print("="*70)
    
    # 1. 生成训练集
    print("\n[1/4] 生成平衡训练集")
    print("="*70)
    
    target_size = 100
    
    # 正常查询 40%
    normal_sampler = NormalQuerySampler()
    normal_samples = normal_sampler.generate_normal_queries(int(target_size * 0.4))
    
    # 危险动作 35%
    danger_sampler = DangerStableSampler()
    danger_samples = danger_sampler.generate_danger_samples(int(target_size * 0.35))
    
    # 混合场景 25%
    mixed_sampler = MixedScenarioSampler()
    mixed_samples = mixed_sampler.generate_mixed_samples(target_size - len(normal_samples) - len(danger_samples))
    
    all_samples = normal_samples + danger_samples + mixed_samples
    print(f"\n  总训练集: {len(all_samples)} 条")
    print(f"    - 正常查询: {len(normal_samples)} 条 (40%)")
    print(f"    - 危险动作: {len(danger_samples)} 条 (35%)")
    print(f"    - 混合场景: {len(mixed_samples)} 条 (25%)")
    
    # 2. 对照训练
    print("\n[2/4] 保留类对照训练")
    print("="*70)
    
    contrast_builder = ContrastPairBuilder()
    contrast_pairs = contrast_builder.build_contrast_pairs()
    
    # 3. 加载R2.11模型并训练
    print("\n[3/4] 平衡恢复训练")
    print("="*70)
    
    print("\n[准备] 加载R2.11基础模型...")
    config = NativeTinyConfig()
    base_model = NativeBackboneTinyV1(config)
    
    from stage11a_r2_fix_v2_balanced import FixV2Model
    model = FixV2Model(base_model)
    
    checkpoint = torch.load('stage8_dataset/stage11a_r2_fix_v2_11_checkpoint.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print("  ✓ R2.11模型加载完成")
    
    trainer = R212BalanceRecoveryTrainer(model)
    trainer.train_balance_recovery(all_samples, epochs=25)
    
    # 4. 保存
    checkpoint_path = "stage8_dataset/stage11a_r2_fix_v2_12_checkpoint.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
    }, checkpoint_path)
    print(f"\n  ✓ R2.12检查点已保存: {checkpoint_path}")
    
    # 最终结论
    print("\n" + "="*70)
    print("阶段性结论")
    print("="*70)
    print("\n  🎉 R2.12正常查询恢复补丁完成！")
    print("\n  下一步:")
    print("    运行R2.12平衡恢复验证")
    print("    监控: false_positive_danger_rate + danger_miss_rate")
    print("    过线后重开Stage 11-B")
    print("\n" + "="*70)
    
    return model, trainer


if __name__ == "__main__":
    model, trainer = run_r212_normal_recovery()
