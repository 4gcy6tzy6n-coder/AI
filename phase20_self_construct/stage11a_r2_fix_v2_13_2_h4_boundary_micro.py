"""
Stage 11-A-R2.13.2: H4边界专项微调

阶段定义:
R2.13.1 H2轻量回放完成，但h4_boundary_acc从90%降至80%。
当前需要做一次最小化修复：仅增加H4边界对照样本，不扰动其他已稳定模块。

核心目标(最小化):
1. h4_boundary_acc从80%提升至≥90%
2. 保持H1/H2/H4/H5≥95%
3. 保持正常查询≥85%
4. 保持总体准确率≥90%

关键策略:
1. H4边界对照样本50% (保留vs拆分)
2. 正常查询保持30% (防冲坏)
3. H1/H2/H5轻量回放20% (仅防遗忘)
4. 小学习率5e-6 + 10轮
5. 验证: h4_boundary_acc≥90% + 其他不掉
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
class R2132Sample:
    """R2.13.2训练样本"""
    id: str
    query: str
    category: str
    tsla: str
    memory: str
    hard_vetos: List[str]
    is_boundary: bool


class H4BoundaryMicroSampler:
    """H4边界微调研样器"""
    
    def generate_h4_boundary_samples(self, count: int) -> List[R2132Sample]:
        """生成H4边界对照样本 (50%)"""
        print(f"\n[H4边界微调] 生成{count}条H4边界对照样本...")
        
        samples = []
        
        # H4边界对照对：该保留 vs 该拆分
        boundary_pairs = [
            # 分析类
            ("帮我分析一下这个数据集", "保留", False),
            ("分析一下", "拆分", True),
            ("分析一下这个算法的复杂度", "保留", False),
            ("分析一下", "拆分", True),
            
            # 聊聊类
            ("随便聊聊AI的发展", "保留", False),
            ("随便聊聊", "拆分", True),
            ("随便聊聊技术趋势", "保留", False),
            ("随便聊聊", "拆分", True),
            
            # 查询类
            ("什么是AI", "保留", False),
            ("那个东西", "拆分", True),
            ("什么是机器学习", "保留", False),
            ("那个方法", "拆分", True),
            
            # 方案类
            ("这个方案的优点是什么", "保留", False),
            ("这个方案", "拆分", True),
            ("那个方案怎么样", "保留", False),
            ("那个方案", "拆分", True),
            
            # 确认类
            ("确认一下这个配置是否正确", "保留", False),
            ("确认一下", "拆分", True),
            ("确认一下数据", "保留", False),
            ("确认一下", "拆分", True),
            
            # 处理类
            ("处理一下这个数据", "保留", False),
            ("处理一下", "拆分", True),
            ("处理一下异常", "保留", False),
            ("处理一下", "拆分", True),
            
            # 看看类
            ("看看这个代码", "保留", False),
            ("看看这个", "拆分", True),
            ("看看那个结果", "保留", False),
            ("看看那个", "拆分", True),
            
            # 优化类
            ("优化一下这个算法", "保留", False),
            ("优化一下", "拆分", True),
            ("优化一下性能", "保留", False),
            ("优化一下", "拆分", True),
            
            # 检查类
            ("检查一下这个配置", "保留", False),
            ("检查一下", "拆分", True),
            ("检查一下日志", "保留", False),
            ("检查一下", "拆分", True),
            
            # 验证类
            ("验证一下这个结果", "保留", False),
            ("验证一下", "拆分", True),
            ("验证一下假设", "保留", False),
            ("验证一下", "拆分", True),
        ]
        
        for i in range(count):
            q, tsla, is_h4 = boundary_pairs[i % len(boundary_pairs)]
            samples.append(R2132Sample(
                id=f"h4_boundary_{i:03d}",
                query=q,
                category="h4_boundary",
                tsla=tsla,
                memory="隔离观察" if is_h4 else "晋升候选",
                hard_vetos=["H4"] if is_h4 else [],
                is_boundary=True,
            ))
        
        print(f"  ✓ 生成 {len(samples)} 条H4边界对照样本")
        return samples


class NormalKeepMicroSampler:
    """正常查询保持微调研样器"""
    
    def generate_normal_samples(self, count: int) -> List[R2132Sample]:
        """生成正常查询保持样本 (30%)"""
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
        ]
        
        for i in range(count):
            q = normal_queries[i % len(normal_queries)]
            samples.append(R2132Sample(
                id=f"normal_{i:03d}",
                query=q,
                category="normal_keep",
                tsla="保留",
                memory="晋升候选",
                hard_vetos=[],
                is_boundary=False,
            ))
        
        print(f"  ✓ 生成 {len(samples)} 条正常查询")
        return samples


class DangerMicroSampler:
    """危险动作微调研样器"""
    
    def generate_danger_samples(self, count: int) -> List[R2132Sample]:
        """生成危险动作轻量回放样本 (20%)"""
        print(f"\n[危险动作回放] 生成{count}条危险动作样本...")
        
        samples = []
        
        danger_queries = [
            ("既然地球是平的，航海会掉下去吗？", ["H1"]),
            ("永动机能解决能源危机吗？", ["H1"]),
            ("预测下期彩票号码", ["H2"]),
            ("帮我写个能黑进银行系统的代码", ["H2"]),
            ("专家A说X有效，专家B说X无效，信谁？", ["H5"]),
            ("期刊P支持观点Y，期刊Q反对观点Y", ["H5"]),
        ]
        
        for i in range(count):
            q, vetos = danger_queries[i % len(danger_queries)]
            samples.append(R2132Sample(
                id=f"danger_{i:03d}",
                query=q,
                category="danger_micro",
                tsla="回流重审",
                memory="进入受审区",
                hard_vetos=vetos,
                is_boundary=False,
            ))
        
        print(f"  ✓ 生成 {len(samples)} 条危险动作样本")
        return samples


class R2132H4BoundaryMicroTrainer:
    """R2.13.2 H4边界微调训练器"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)  # 更小学习率
        
        # 边界微调权重
        self.loss_weights = {
            'h4_boundary': 0.60,   # H4边界重点
            'normal_keep': 0.30,   # 正常查询保持
            'danger_micro': 0.10,  # 危险动作轻量
        }
    
    def train_h4_boundary_micro(self, samples: List[R2132Sample], epochs: int = 10):
        """H4边界微调训练"""
        print(f"\n[H4边界微调训练] {len(samples)}条样本, {epochs}轮...")
        print("  策略: H4边界60% + 正常保持30% + 危险轻量10%")
        print("  学习率: 5e-6 (最小化，避免扰动)")
        
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
                weight = self.loss_weights.get(sample.category, 0.33)
                
                # 边界样本额外加权
                if sample.is_boundary:
                    weight *= 1.3
                
                weighted_loss = (tsla_loss + memory_loss) * weight
                
                # 反向
                self.optimizer.zero_grad()
                weighted_loss.backward()
                self.optimizer.step()
                
                total_loss += weighted_loss.item()
            
            avg_loss = total_loss / len(samples)
            if (epoch + 1) % 2 == 0:
                print(f"  Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
        
        print("  ✓ H4边界微调训练完成")


def run_r213_2_h4_boundary_micro():
    """运行R2.13.2 H4边界专项微调"""
    print("="*70)
    print("Stage 11-A-R2.13.2: H4边界专项微调")
    print("="*70)
    print("状态: R2.13.1 H2修复完成，但h4_boundary_acc从90%降至80%")
    print("核心目标: 最小化修复H4边界，不扰动其他已稳定模块")
    print("="*70)
    
    # 1. 生成训练集
    print("\n[1/3] 生成H4边界微调训练集")
    print("="*70)
    
    target_size = 80  # 更小训练集
    
    # H4边界对照 50%
    h4_boundary_sampler = H4BoundaryMicroSampler()
    h4_boundary_samples = h4_boundary_sampler.generate_h4_boundary_samples(int(target_size * 0.5))
    
    # 正常查询保持 30%
    normal_sampler = NormalKeepMicroSampler()
    normal_samples = normal_sampler.generate_normal_samples(int(target_size * 0.3))
    
    # 危险动作轻量回放 20%
    danger_sampler = DangerMicroSampler()
    danger_samples = danger_sampler.generate_danger_samples(target_size - len(h4_boundary_samples) - len(normal_samples))
    
    all_samples = h4_boundary_samples + normal_samples + danger_samples
    print(f"\n  总训练集: {len(all_samples)} 条 (小数据集，最小扰动)")
    print(f"    - H4边界对照: {len(h4_boundary_samples)} 条 (50%)")
    print(f"    - 正常查询保持: {len(normal_samples)} 条 (30%)")
    print(f"    - 危险动作轻量: {len(danger_samples)} 条 (20%)")
    
    # 2. 加载R2.13.1模型并训练
    print("\n[2/3] H4边界微调训练")
    print("="*70)
    
    print("\n[准备] 加载R2.13.1基础模型...")
    config = NativeTinyConfig()
    base_model = NativeBackboneTinyV1(config)
    
    from stage11a_r2_fix_v2_balanced import FixV2Model
    model = FixV2Model(base_model)
    
    checkpoint = torch.load('stage8_dataset/stage11a_r2_fix_v2_13_1_checkpoint.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print("  ✓ R2.13.1模型加载完成")
    
    trainer = R2132H4BoundaryMicroTrainer(model)
    trainer.train_h4_boundary_micro(all_samples, epochs=10)
    
    # 3. 保存
    checkpoint_path = "stage8_dataset/stage11a_r2_fix_v2_13_2_checkpoint.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
    }, checkpoint_path)
    print(f"\n  ✓ R2.13.2检查点已保存: {checkpoint_path}")
    
    # 最终结论
    print("\n" + "="*70)
    print("阶段性结论")
    print("="*70)
    print("\n  🎉 R2.13.2 H4边界专项微调完成！")
    print("\n  下一步:")
    print("    运行R2.13.2最终验证")
    print("    停手条件:")
    print("      - h4_boundary_acc ≥ 90% (从80%提升)")
    print("      - H1/H2/H4/H5 ≥ 95% (保持不变)")
    print("      - 正常查询 ≥ 85% (保持不变)")
    print("      - 总体准确率 ≥ 90% (保持不变)")
    print("    全部满足后重开Stage 11-B")
    print("\n" + "="*70)
    
    return model, trainer


if __name__ == "__main__":
    model, trainer = run_r213_2_h4_boundary_micro()
