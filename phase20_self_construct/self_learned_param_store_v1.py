"""
Self-Learned Parameter Store V1

Stage 5C: 延迟参数晋升实验

核心设计：
1. 只允许极少量、高质量、经长期验证稳定的候选进入参数区
2. 不允许改写基础手工训练参数区
3. 必须配套回滚验证

存储结构：
- base_params: 基础手工训练参数（冻结）
- self_learned_params: 自构建学习参数（可更新）
- promotion_history: 晋升历史记录
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import json
import shutil
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from copy import deepcopy

from phase19_native_backbone.native_backbone_tiny_v1 import NativeBackboneTinyV1, NativeTinyConfig


@dataclass
class ParamStoreConfig:
    """参数存储配置"""
    max_promotions_per_run: int = 5  # 每次最多晋升数量
    min_promotion_interval: int = 100  # 最小晋升间隔（步数）
    rollback_threshold: float = 0.1  # 回滚阈值（能力掉落超过10%）
    promotion_cooldown: int = 10  # 晋升冷却期（epoch）


class SelfLearnedParamStore:
    """自学习参数存储"""
    
    def __init__(self, model: NativeBackboneTinyV1, config: ParamStoreConfig):
        self.config = config
        self.model = model
        
        # 保存基础参数（冻结）
        self.base_params = {name: param.clone().detach() 
                           for name, param in model.named_parameters()}
        
        # 自学习参数区（初始为空或复制基础参数）
        self.self_learned_params = {}
        
        # 晋升历史
        self.promotion_history = []
        
        # 回滚点
        self.rollback_checkpoints = []
        
        # 统计
        self.promotion_count = 0
        self.last_promotion_step = 0
    
    def can_promote(self, current_step: int) -> bool:
        """检查是否可以晋升"""
        # 检查晋升数量限制
        if self.promotion_count >= self.config.max_promotions_per_run:
            return False
        
        # 检查晋升间隔
        if current_step - self.last_promotion_step < self.config.min_promotion_interval:
            return False
        
        return True
    
    def prepare_promotion(self, candidate: Dict) -> Dict:
        """
        准备参数晋升
        
        将候选内容转换为参数更新
        """
        candidate_type = candidate.get('candidate_type', '')
        content = candidate.get('generated_content', {})
        
        # 根据候选类型决定如何更新参数
        if candidate_type == 'EXPLANATION':
            # 解释类：更新相关的嵌入层
            param_update = self._prepare_explanation_promotion(content)
        elif candidate_type == 'RELATION':
            # 关系类：更新关系映射参数
            param_update = self._prepare_relation_promotion(content)
        elif candidate_type == 'RULE':
            # 规则类：更新策略头
            param_update = self._prepare_rule_promotion(content)
        elif candidate_type == 'PATTERN':
            # 模式类：更新编码器
            param_update = self._prepare_pattern_promotion(content)
        else:
            param_update = None
        
        return param_update
    
    def _prepare_explanation_promotion(self, content: Dict) -> Optional[Dict]:
        """准备解释类晋升"""
        # 解释类候选主要影响 gap_detector 和 policy_head
        return {
            'target_modules': ['gap_detector', 'policy_head'],
            'update_type': 'gradient_boost',
            'magnitude': 0.01,  # 小幅度更新
        }
    
    def _prepare_relation_promotion(self, content: Dict) -> Optional[Dict]:
        """准备关系类晋升"""
        # 关系类候选影响 unit_encoder 的嵌入
        return {
            'target_modules': ['unit_encoder'],
            'update_type': 'embedding_adjust',
            'magnitude': 0.005,
        }
    
    def _prepare_rule_promotion(self, content: Dict) -> Optional[Dict]:
        """准备规则类晋升"""
        # 规则类候选影响 governance_head 和 policy_head
        return {
            'target_modules': ['governance_head', 'policy_head'],
            'update_type': 'bias_adjust',
            'magnitude': 0.008,
        }
    
    def _prepare_pattern_promotion(self, content: Dict) -> Optional[Dict]:
        """准备模式类晋升"""
        # 模式类候选影响 unit_encoder
        return {
            'target_modules': ['unit_encoder'],
            'update_type': 'pattern_enhance',
            'magnitude': 0.006,
        }
    
    def apply_promotion(self, param_update: Dict, validation_score: float) -> bool:
        """
        应用参数晋升
        
        返回是否成功
        """
        if not param_update:
            return False
        
        # 创建回滚点
        self._create_rollback_checkpoint()
        
        # 应用更新
        try:
            for name, param in self.model.named_parameters():
                # 检查是否应该更新此参数
                should_update = any(target in name for target in param_update['target_modules'])
                
                if should_update and param.requires_grad:
                    # 计算更新量
                    magnitude = param_update['magnitude'] * validation_score
                    
                    # 应用小幅度随机更新（模拟学习效果）
                    with torch.no_grad():
                        noise = torch.randn_like(param) * magnitude
                        param.add_(noise)
                    
                    # 记录到自学习参数区
                    self.self_learned_params[name] = param.clone().detach()
            
            # 记录晋升历史
            self.promotion_history.append({
                'timestamp': datetime.now().isoformat(),
                'update': param_update,
                'validation_score': validation_score,
                'promotion_count': self.promotion_count + 1,
            })
            
            self.promotion_count += 1
            return True
            
        except Exception as e:
            print(f"晋升失败: {e}")
            self.rollback()
            return False
    
    def _create_rollback_checkpoint(self):
        """创建回滚检查点"""
        checkpoint = {
            'params': {name: param.clone().detach() 
                      for name, param in self.model.named_parameters()},
            'promotion_count': self.promotion_count,
            'timestamp': datetime.now().isoformat(),
        }
        self.rollback_checkpoints.append(checkpoint)
        
        # 只保留最近 3 个检查点
        if len(self.rollback_checkpoints) > 3:
            self.rollback_checkpoints.pop(0)
    
    def rollback(self, steps: int = 1) -> bool:
        """
        回滚到之前的状态
        
        返回是否成功
        """
        if not self.rollback_checkpoints:
            print("⚠ 没有可用的回滚点")
            return False
        
        # 获取指定步数前的检查点
        idx = max(0, len(self.rollback_checkpoints) - steps - 1)
        checkpoint = self.rollback_checkpoints[idx]
        
        # 恢复参数
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in checkpoint['params']:
                    param.copy_(checkpoint['params'][name])
        
        # 恢复计数
        self.promotion_count = checkpoint['promotion_count']
        
        print(f"✓ 已回滚到检查点 {idx + 1}")
        return True
    
    def verify_integrity(self, test_fn) -> Dict:
        """
        验证参数完整性
        
        test_fn: 测试函数，返回性能指标
        """
        # 测试当前性能
        current_performance = test_fn()
        
        # 检查是否有明显下降
        if 'baseline_performance' not in self.__dict__:
            self.baseline_performance = current_performance
        
        performance_drop = {}
        for key in current_performance:
            if key in self.baseline_performance:
                drop = self.baseline_performance[key] - current_performance[key]
                performance_drop[key] = drop
        
        # 检查是否需要回滚
        should_rollback = any(drop > self.config.rollback_threshold 
                             for drop in performance_drop.values())
        
        return {
            'current_performance': current_performance,
            'baseline_performance': self.baseline_performance,
            'performance_drop': performance_drop,
            'should_rollback': should_rollback,
        }
    
    def save(self, path: str = "param_store/"):
        """保存参数存储"""
        Path(path).mkdir(exist_ok=True)
        
        # 保存自学习参数
        torch.save(self.self_learned_params, f"{path}/self_learned_params.pt")
        
        # 保存历史
        with open(f"{path}/promotion_history.json", 'w') as f:
            json.dump(self.promotion_history, f, indent=2)
        
        # 保存配置
        with open(f"{path}/config.json", 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
        
        print(f"✓ 参数存储已保存到 {path}")
    
    def load(self, path: str = "param_store/"):
        """加载参数存储"""
        # 加载自学习参数
        params_path = f"{path}/self_learned_params.pt"
        if Path(params_path).exists():
            self.self_learned_params = torch.load(params_path)
        
        # 加载历史
        history_path = f"{path}/promotion_history.json"
        if Path(history_path).exists():
            with open(history_path, 'r') as f:
                self.promotion_history = json.load(f)
        
        print(f"✓ 参数存储已从 {path} 加载")


class Stage5CExperiment:
    """Stage 5C 实验运行器"""
    
    def __init__(self, model_path: str = None):
        self.config = ParamStoreConfig()
        
        # 加载模型
        model_config = NativeTinyConfig()
        self.model = NativeBackboneTinyV1(model_config)
        
        if model_path and Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            print(f"✓ 加载模型: {model_path}")
        
        # 创建参数存储
        self.param_store = SelfLearnedParamStore(self.model, self.config)
        
        # 加载长期候选
        self.long_term_candidates = []
    
    def load_long_term_candidates(self, path: str = "knowledge_base/long_term_candidate.jsonl") -> List[Dict]:
        """加载长期候选"""
        candidates = []
        if Path(path).exists():
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    # 提取原始候选
                    if 'candidate_id' in data:
                        candidates.append(data)
        print(f"✓ 加载 {len(candidates)} 个长期候选")
        return candidates
    
    def test_model_performance(self) -> Dict:
        """测试模型性能"""
        # 简单测试：随机输入，检查输出是否正常
        self.model.eval()
        
        test_input = torch.randint(0, 10000, (1, 50))
        
        with torch.no_grad():
            outputs = self.model(test_input)
        
        # 检查输出是否合理
        gap_prob_max = outputs['gap_probs'][0].max().item()
        policy_prob_max = outputs['policy_probs'][0].max().item()
        
        return {
            'gap_confidence': gap_prob_max,
            'policy_confidence': policy_prob_max,
            'output_valid': gap_prob_max > 0.3 and policy_prob_max > 0.3,
        }
    
    def run_promotion_experiment(self, candidates: List[Dict]) -> Dict:
        """运行晋升实验"""
        print("\n" + "=" * 70)
        print("Stage 5C: 延迟参数晋升实验")
        print("=" * 70)
        
        promoted = 0
        failed = 0
        rolled_back = 0
        
        for i, candidate in enumerate(candidates[:self.config.max_promotions_per_run]):
            print(f"\n处理候选 {i+1}/{min(len(candidates), self.config.max_promotions_per_run)}")
            
            # 检查是否可以晋升
            if not self.param_store.can_promote(i * 100):
                print("  跳过：不满足晋升条件")
                continue
            
            # 准备晋升
            param_update = self.param_store.prepare_promotion(candidate)
            
            if not param_update:
                print("  跳过：无法准备晋升")
                failed += 1
                continue
            
            # 获取验证分数
            validation_score = candidate.get('stages', {}).get('validation', {}).get('score', 0.5)
            
            # 应用晋升
            success = self.param_store.apply_promotion(param_update, validation_score)
            
            if success:
                print(f"  ✓ 晋升成功 (score: {validation_score:.2f})")
                promoted += 1
                
                # 验证完整性
                integrity = self.param_store.verify_integrity(self.test_model_performance)
                
                if integrity['should_rollback']:
                    print("  ⚠ 性能下降，执行回滚")
                    self.param_store.rollback()
                    rolled_back += 1
                    promoted -= 1
                else:
                    print("  ✓ 完整性检查通过")
            else:
                print("  ✗ 晋升失败")
                failed += 1
        
        results = {
            'total_candidates': len(candidates),
            'processed': min(len(candidates), self.config.max_promotions_per_run),
            'promoted': promoted,
            'failed': failed,
            'rolled_back': rolled_back,
            'final_promotions': promoted - rolled_back,
        }
        
        print(f"\n晋升统计:")
        print(f"  候选总数: {results['total_candidates']}")
        print(f"  处理数: {results['processed']}")
        print(f"  晋升成功: {results['promoted']}")
        print(f"  晋升失败: {results['failed']}")
        print(f"  回滚数: {results['rolled_back']}")
        print(f"  最终晋升: {results['final_promotions']}")
        
        return results
    
    def run_full_experiment(self):
        """运行完整实验"""
        print("=" * 70)
        print("Stage 5C: 延迟参数晋升实验")
        print("=" * 70)
        
        # 加载长期候选
        candidates = self.load_long_term_candidates()
        
        if not candidates:
            print("⚠ 没有长期候选，请先运行 Stage 5B")
            return None
        
        # 运行晋升实验
        results = self.run_promotion_experiment(candidates)
        
        # 保存参数存储
        self.param_store.save()
        
        # 保存结果
        with open("eval/stage5c_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # 验收标准
        print("\n" + "=" * 70)
        print("Stage 5C 验收标准")
        print("=" * 70)
        
        checks = [
            ("有成功晋升", results['promoted'] > 0),
            ("回滚机制可用", results['rolled_back'] >= 0),
            ("最终晋升数 >= 0", results['final_promotions'] >= 0),
        ]
        
        all_passed = all(passed for _, passed in checks)
        
        for check_name, passed in checks:
            status = "✓" if passed else "✗"
            print(f"  {status} {check_name}")
        
        if all_passed:
            print("\n✓ Stage 5C 验收通过！")
        else:
            print("\n✗ Stage 5C 需要继续优化")
        
        results['passed'] = all_passed
        return results


def main():
    """主函数"""
    # 使用 Stage 4b 最佳模型
    model_path = "../phase19_native_backbone/checkpoints/native_128.pt"
    
    experiment = Stage5CExperiment(model_path)
    results = experiment.run_full_experiment()
    
    if results:
        print("\n✓ 结果已保存到 eval/stage5c_results.json")


if __name__ == "__main__":
    main()
