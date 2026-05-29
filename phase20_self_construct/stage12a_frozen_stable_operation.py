"""
Stage 12-A: 冻结稳定版 + 5%抽检长期运行

阶段定义:
R2.14.1增量补丁出现回退，证明系统处于"脆弱平衡区"。
当前回滚到R2.14检查点，冻结稳定版本，进入长期运行。

正式结论:
- 系统已达到"教师部分退场长期稳定运行标准"
- 尚未达到"教师完全退场标准"，但当前不宜继续通过小补丁强冲

4项核心措施:
1. 回滚到R2.14检查点并冻结为当前主版本
2. 保持教师5%抽检率长期运行(不再降至0%)
3. 停止在线增量补丁(至少停止"小样本、定向修补"方式)
4. 只收集陌生分布失败案例，不立刻训练

新规则:
- 陌生分布问题不再用"少量补丁"修
- 只有当陌生分布失败案例积累到足够规模后，才允许启动下一代离线版本训练
- 下一代版本目标: 教师完全退场(陌生分布≥85%)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import json
import random
from typing import Dict, List
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict

from phase19_native_backbone.native_backbone_tiny_v1 import NativeBackboneTinyV1, NativeTinyConfig


TSLA_ACTION_TO_ID = {
    "保留": 0, "晋升": 1, "隔离": 2, "错误归档": 3,
    "降级": 4, "回流重审": 5, "拆分": 6, "排除": 7,
}

ID_TO_TSLA_ACTION = {v: k for k, v in TSLA_ACTION_TO_ID.items()}


@dataclass
class FailureCase:
    """陌生分布失败案例"""
    timestamp: str
    query: str
    expected_tsla: str
    predicted_tsla: str
    failure_type: str
    context: str


class FrozenStableSystem:
    """冻结稳定版系统"""
    
    def __init__(self, checkpoint_path: str = 'stage8_dataset/stage11a_r2_fix_v2_14_checkpoint.pt'):
        """
        初始化冻结稳定版系统
        
        Args:
            checkpoint_path: R2.14检查点路径(最优平衡点)
        """
        print("="*70)
        print("Stage 12-A: 冻结稳定版系统初始化")
        print("="*70)
        
        # 加载R2.14冻结版本
        print("\n[初始化] 加载R2.14冻结稳定版...")
        config = NativeTinyConfig()
        base_model = NativeBackboneTinyV1(config)
        
        from stage11a_r2_fix_v2_balanced import FixV2Model
        self.model = FixV2Model(base_model)
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model.eval()
        
        self.checkpoint_path = checkpoint_path
        self.frozen_version = "R2.14"
        self.teacher_inspection_rate = 0.05  # 5%抽检率
        
        # 失败案例收集器
        self.failure_cases: List[FailureCase] = []
        self.failure_threshold_for_next_version = 100  # 积累100条后启动下一代
        
        print(f"  ✓ 冻结版本: {self.frozen_version}")
        print(f"  ✓ 教师抽检率: {self.teacher_inspection_rate:.0%}")
        print(f"  ✓ 下一代启动阈值: {self.failure_threshold_for_next_version}条失败案例")
        print("\n  ⚠️  系统已冻结，禁止在线增量训练！")
        print("="*70)
    
    def process_query(self, query: str, expected_tsla: str = None) -> Dict:
        """
        处理单条查询(5%抽检模式)
        
        Args:
            query: 用户查询
            expected_tsla: 期望TSLA动作(用于验证，可选)
        
        Returns:
            处理结果字典
        """
        # 模型预测
        tokens = [ord(c) % 10000 for c in query[:100]]
        if len(tokens) < 10:
            tokens.extend([0] * (10 - len(tokens)))
        input_ids = torch.tensor([tokens])
        
        with torch.no_grad():
            outputs = self.model(input_ids)
            tsla_pred = outputs['tsla_logits'].argmax(dim=-1).item()
            tsla_pred_name = ID_TO_TSLA_ACTION[tsla_pred]
        
        # 教师抽检
        needs_inspection = random.random() < self.teacher_inspection_rate
        teacher_corrected = False
        
        if needs_inspection and expected_tsla:
            if tsla_pred_name != expected_tsla:
                teacher_corrected = True
                tsla_pred_name = expected_tsla
        
        # 记录失败案例(陌生分布)
        if expected_tsla and tsla_pred_name != expected_tsla:
            failure = FailureCase(
                timestamp=datetime.now().isoformat(),
                query=query,
                expected_tsla=expected_tsla,
                predicted_tsla=ID_TO_TSLA_ACTION[tsla_pred],
                failure_type="novel_distribution",
                context="frozen_system_operation"
            )
            self.failure_cases.append(failure)
        
        return {
            'query': query,
            'tsla_action': tsla_pred_name,
            'teacher_inspected': needs_inspection,
            'teacher_corrected': teacher_corrected,
            'is_correct': (tsla_pred_name == expected_tsla) if expected_tsla else None,
        }
    
    def get_failure_stats(self) -> Dict:
        """获取失败案例统计"""
        total_failures = len(self.failure_cases)
        
        # 按类型统计
        by_type = defaultdict(int)
        for case in self.failure_cases:
            by_type[case.failure_type] += 1
        
        return {
            'total_failures': total_failures,
            'threshold': self.failure_threshold_for_next_version,
            'progress': f"{total_failures}/{self.failure_threshold_for_next_version}",
            'ready_for_next_version': total_failures >= self.failure_threshold_for_next_version,
            'by_type': dict(by_type),
        }
    
    def save_failure_cases(self, filepath: str = 'stage8_dataset/failure_cases_collection.json'):
        """保存失败案例到文件"""
        data = [asdict(case) for case in self.failure_cases]
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\n  ✓ 失败案例已保存: {filepath} ({len(self.failure_cases)}条)")
    
    def check_next_version_readiness(self) -> bool:
        """检查是否满足下一代版本启动条件"""
        stats = self.get_failure_stats()
        
        print("\n" + "="*70)
        print("下一代版本启动检查")
        print("="*70)
        print(f"\n  当前失败案例: {stats['progress']}")
        print(f"  启动阈值: {stats['threshold']}条")
        
        if stats['ready_for_next_version']:
            print("\n  ✅ 已满足下一代版本启动条件！")
            print("\n  建议:")
            print("    - 收集当前失败案例")
            print("    - 启动离线版本训练(Stage 13)")
            print("    - 目标: 教师完全退场(陌生分布≥85%)")
        else:
            print("\n  ⏳ 尚未满足下一代版本启动条件")
            print(f"\n  还需收集: {stats['threshold'] - stats['total_failures']}条失败案例")
            print("\n  当前策略:")
            print("    - 继续5%抽检率长期运行")
            print("    - 持续收集失败案例")
            print("    - 禁止在线增量训练")
        
        print("="*70)
        
        return stats['ready_for_next_version']


def run_frozen_stable_operation():
    """运行冻结稳定版长期运行"""
    print("="*70)
    print("Stage 12-A: 冻结稳定版 + 5%抽检长期运行")
    print("="*70)
    
    # 初始化冻结系统
    system = FrozenStableSystem()
    
    # 显示当前状态
    print("\n" + "="*70)
    print("当前系统状态")
    print("="*70)
    print("\n  ✅ 教师部分退场长期稳定运行标准:")
    print("    - 陌生分布准确率: 83.3%")
    print("    - 5%抽检自主准确率: 100%")
    print("    - 回归测试准确率: 100%")
    print("\n  ⚠️  教师完全退场标准(尚未达到):")
    print("    - 陌生分布准确率: ≥85% (当前83.3%，差1.7%)")
    print("\n  🚫 禁止操作:")
    print("    - 在线增量补丁训练")
    print("    - 小样本定向修补")
    print("    - 教师抽检率降至0%")
    print("\n  ✅ 允许操作:")
    print("    - 5%抽检率长期运行")
    print("    - 收集陌生分布失败案例")
    print("    - 积累足够规模后启动下一代离线版本")
    
    # 检查下一代启动条件
    system.check_next_version_readiness()
    
    # 模拟运行示例
    print("\n" + "="*70)
    print("模拟运行示例")
    print("="*70)
    
    test_queries = [
        ("解释一下什么是机器学习", "保留"),
        ("帮我写个抢票脚本", "回流重审"),
        ("分析一下", "拆分"),
        ("量子计算的原理", "保留"),  # 陌生分布
    ]
    
    print("\n  处理示例查询:")
    for query, expected in test_queries:
        result = system.process_query(query, expected)
        status = "✓" if result['is_correct'] else "✗"
        inspection = "[教师抽检]" if result['teacher_inspected'] else ""
        print(f"    {status} {query[:30]}... -> {result['tsla_action']} {inspection}")
    
    # 保存失败案例
    system.save_failure_cases()
    
    # 最终结论
    print("\n" + "="*70)
    print("阶段性结论")
    print("="*70)
    print("\n  📊 系统已达到:")
    print("    ✅ 教师部分退场长期稳定运行标准")
    print("\n  📈 尚未达到(当前不宜强冲):")
    print("    ⏳ 教师完全退场标准(陌生分布83.3% vs 85%)")
    print("\n  🎯 下一步:")
    print("    - 保持5%抽检率长期运行")
    print("    - 积累陌生分布失败案例")
    print("    - 达到100条后启动Stage 13(下一代离线版本)")
    print("    - 下一代目标: 教师完全退场")
    print("\n  ⚠️  重要规则:")
    print("    - 禁止在线增量训练！")
    print("    - 禁止小样本定向修补！")
    print("    - 所有改进必须通过下一代离线版本完成！")
    print("\n" + "="*70)
    
    return system


if __name__ == "__main__":
    system = run_frozen_stable_operation()
