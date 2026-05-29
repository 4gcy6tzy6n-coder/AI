"""
Stage 6 Smoke Test

全链路整合冒烟测试 - 验证核心功能在整合系统中的可用性

测试范围:
1. 单步晋升 (1-step promotion)
2. 两步连续晋升 (2-step continuous promotion)
3. 回滚机制 (rollback mechanism)

验收标准:
- 1 步整合运行正常
- 2 步整合运行正常
- rollback 功能正常
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import json
from typing import Dict, List
from datetime import datetime

from stage6_runtime_orchestrator import (
    Stage6Orchestrator,
    Stage6Config,
    PromotionResult,
    StepMetrics,
)


# ==================== 测试配置 ====================

SMOKE_TEST_CONFIG = {
    # 使用 6B 配置作为默认
    'base_kl_weights': {
        'gap': 0.30,
        'policy': 0.30,
        'governance': 0.30,
        'writeback': 0.38,
    },
    'learning_rate': 1.0e-5,
    'replay_ratio': 0.35,
    'step1_max_change': 0.003,
    'step2_max_change': 0.008,
    'old_ability_threshold': 0.10,  # 回滚阈值
    'writeback_threshold': 0.05,
    'auto_rollback': True,
    # 降低阈值确保候选能进入参数晋升路径
    'param_promotion_threshold': 0.80,  # 默认 0.90
    'kb_promotion_threshold': 0.70,     # 默认 0.80
}


# ==================== 测试工具 ====================

class SmokeTestReporter:
    """冒烟测试报告器"""
    
    def __init__(self):
        self.tests: List[Dict] = []
        self.passed = 0
        self.failed = 0
    
    def add_test(self, name: str, passed: bool, details: Dict = None):
        """添加测试结果"""
        test_result = {
            'name': name,
            'passed': passed,
            'timestamp': datetime.now().isoformat(),
            'details': details or {},
        }
        self.tests.append(test_result)
        if passed:
            self.passed += 1
        else:
            self.failed += 1
    
    def print_summary(self):
        """打印测试摘要"""
        print("\n" + "="*70)
        print("Stage 6 Smoke Test - 结果摘要")
        print("="*70)
        
        for test in self.tests:
            status = "✓ PASS" if test['passed'] else "✗ FAIL"
            print(f"{status}: {test['name']}")
            if test['details']:
                for key, value in test['details'].items():
                    print(f"    {key}: {value}")
        
        print("-"*70)
        print(f"总计: {len(self.tests)} 个测试")
        print(f"通过: {self.passed}")
        print(f"失败: {self.failed}")
        
        if self.failed == 0:
            print("\n🎉 所有冒烟测试通过! Stage 6 全链路整合可用。")
        else:
            print(f"\n⚠️ 有 {self.failed} 个测试失败，需要修复。")
        
        print("="*70)
    
    def export(self, filepath: str):
        """导出测试报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total': len(self.tests),
            'passed': self.passed,
            'failed': self.failed,
            'tests': self.tests,
        }
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n测试报告已导出: {filepath}")


# ==================== 测试用例 ====================

def test_single_step_promotion() -> Dict:
    """
    测试 1: 单步整合运行
    
    验证目标:
    - 完整链路能正常执行
    - 指标收集正常
    - 结果符合预期格式
    """
    print("\n" + "-"*70)
    print("测试 1: 单步整合运行")
    print("-"*70)
    
    try:
        # 创建编排器
        config = Stage6Config(**SMOKE_TEST_CONFIG)
        orchestrator = Stage6Orchestrator(config)
        
        # 使用固定的随机种子确保主干输出稳定 (gap=1 以生成候选)
        import torch
        torch.manual_seed(42)
        
        # 执行单步晋升
        result = orchestrator.run_single_step("测试查询: AI 和机器学习的关系")
        
        # 验证结果
        checks = {
            'success': result.success,
            'has_target_improvement': isinstance(result.target_improvement, float),
            'has_old_ability_drop': isinstance(result.old_ability_drop, float),
            'has_writeback_change': isinstance(result.writeback_change, float),
            'has_steps': len(result.steps) > 0,
            'has_kl_weights': bool(result.final_kl_weights),
        }
        
        all_passed = all(checks.values())
        
        details = {
            'success': result.success,
            'target_improvement': f"{result.target_improvement:+.2%}",
            'old_ability_drop': f"{result.old_ability_drop:.2%}",
            'writeback_change': f"{result.writeback_change:+.2%}",
            'steps_count': len(result.steps),
            'rollback_performed': result.rollback_performed,
        }
        
        if all_passed:
            print("✓ 单步整合运行正常")
            print(f"  - 目标提升: {details['target_improvement']}")
            print(f"  - 旧能力掉落: {details['old_ability_drop']}")
            print(f"  - writeback: {details['writeback_change']}")
        else:
            print("✗ 单步整合运行异常")
            for check, passed in checks.items():
                if not passed:
                    print(f"  - 失败: {check}")
        
        return {
            'passed': all_passed,
            'details': details,
        }
        
    except Exception as e:
        print(f"✗ 单步整合运行异常: {e}")
        return {
            'passed': False,
            'details': {'error': str(e)},
        }


def test_two_step_promotion() -> Dict:
    """
    测试 2: 两步连续整合运行
    
    验证目标:
    - 连续两步能正常执行
    - 累积指标收集正常
    - 第二步不会导致过度损伤
    """
    print("\n" + "-"*70)
    print("测试 2: 两步连续整合运行")
    print("-"*70)
    
    try:
        # 创建编排器
        config = Stage6Config(**SMOKE_TEST_CONFIG)
        orchestrator = Stage6Orchestrator(config)
        
        # 使用固定的随机种子确保主干输出稳定
        import torch
        torch.manual_seed(42)
        
        # 执行两步晋升
        contexts = [
            "查询 1: 深度学习框架对比",
            "查询 2: 神经网络优化算法",
        ]
        results = orchestrator.run_continuous_promotion(contexts)
        
        # 验证结果
        checks = {
            'two_results': len(results) == 2,
            'first_success': results[0].success if len(results) > 0 else False,
            'second_success': results[1].success if len(results) > 1 else False,
            'metrics_collected': len(orchestrator.metrics_history) >= 2,
        }
        
        all_passed = all(checks.values())
        
        # 获取累积指标
        report = orchestrator.get_metrics_report()
        
        details = {
            'steps_executed': len(results),
            'metrics_count': len(orchestrator.metrics_history),
            'avg_target_improvement': f"{report.get('avg_target_improvement', 0):+.2%}",
            'max_old_ability_drop': f"{report.get('max_old_ability_drop', 0):.2%}",
        }
        
        if all_passed:
            print("✓ 两步连续整合运行正常")
            print(f"  - 执行步数: {details['steps_executed']}")
            print(f"  - 指标收集: {details['metrics_count']}")
            print(f"  - 平均目标提升: {details['avg_target_improvement']}")
            print(f"  - 最大旧能力掉落: {details['max_old_ability_drop']}")
        else:
            print("✗ 两步连续整合运行异常")
            for check, passed in checks.items():
                if not passed:
                    print(f"  - 失败: {check}")
        
        return {
            'passed': all_passed,
            'details': details,
        }
        
    except Exception as e:
        print(f"✗ 两步连续整合运行异常: {e}")
        return {
            'passed': False,
            'details': {'error': str(e)},
        }


def test_rollback_mechanism() -> Dict:
    """
    测试 3: 回滚机制验证
    
    验证目标:
    - 基线保存正常
    - 回滚触发条件检测正常
    - 回滚执行正常
    """
    print("\n" + "-"*70)
    print("测试 3: 回滚机制验证")
    print("-"*70)
    
    try:
        from stage6_runtime_orchestrator import RollbackHook, NativeBackboneTinyV1, NativeTinyConfig
        
        # 创建配置 (降低阈值以便触发回滚)
        config = Stage6Config(**SMOKE_TEST_CONFIG)
        config.old_ability_threshold = 0.05  # 降低阈值
        config.writeback_threshold = 0.03
        
        # 创建回滚钩子
        rollback_hook = RollbackHook(config)
        
        # 创建模型
        model_config = NativeTinyConfig()
        model = NativeBackboneTinyV1(model_config)
        
        # 保存基线状态
        import copy
        baseline_state = copy.deepcopy(model.state_dict())
        baseline_abilities = {'target': 0.5, 'retrieval': 0.5, 'writeback': 0.5}
        
        rollback_hook.save_baseline(model, baseline_abilities)
        
        # 测试 1: 正常情况不应触发回滚
        normal_metrics = {
            'old_ability_drop': 0.03,  # 低于阈值
            'writeback_change': 0.01,
        }
        should_not_rollback = not rollback_hook.check_need_rollback(normal_metrics)
        
        # 测试 2: 旧能力掉落过高应触发回滚
        high_drop_metrics = {
            'old_ability_drop': 0.08,  # 高于阈值 0.05
            'writeback_change': 0.01,
        }
        should_rollback_old = rollback_hook.check_need_rollback(high_drop_metrics)
        
        # 测试 3: writeback 掉落过高应触发回滚
        high_writeback_metrics = {
            'old_ability_drop': 0.03,
            'writeback_change': 0.05,  # 高于阈值 0.03
        }
        should_rollback_writeback = rollback_hook.check_need_rollback(high_writeback_metrics)
        
        # 测试 4: 回滚执行
        # 修改模型参数
        for param in model.parameters():
            param.data += 0.1
        
        # 执行回滚
        rollback_success = rollback_hook.rollback(model)
        
        # 验证回滚后状态
        state_match = all(
            torch.allclose(baseline_state[k], model.state_dict()[k])
            for k in baseline_state.keys()
        )
        
        checks = {
            'normal_no_rollback': should_not_rollback,
            'old_drop_triggers_rollback': should_rollback_old,
            'writeback_drop_triggers_rollback': should_rollback_writeback,
            'rollback_execution': rollback_success,
            'state_restored': state_match,
        }
        
        all_passed = all(checks.values())
        
        details = {
            'normal_case_ok': should_not_rollback,
            'old_drop_detection': should_rollback_old,
            'writeback_detection': should_rollback_writeback,
            'rollback_success': rollback_success,
            'state_restored': state_match,
        }
        
        if all_passed:
            print("✓ 回滚机制验证通过")
            print(f"  - 正常情况不触发: {details['normal_case_ok']}")
            print(f"  - 旧能力掉落检测: {details['old_drop_detection']}")
            print(f"  - writeback 检测: {details['writeback_detection']}")
            print(f"  - 回滚执行成功: {details['rollback_success']}")
            print(f"  - 状态恢复正确: {details['state_restored']}")
        else:
            print("✗ 回滚机制验证失败")
            for check, passed in checks.items():
                if not passed:
                    print(f"  - 失败: {check}")
        
        return {
            'passed': all_passed,
            'details': details,
        }
        
    except Exception as e:
        print(f"✗ 回滚机制验证异常: {e}")
        import traceback
        traceback.print_exc()
        return {
            'passed': False,
            'details': {'error': str(e)},
        }


def test_integration_chain() -> Dict:
    """
    测试 4: 完整链路验证
    
    验证目标:
    - UnitEncoder → Gap/Policy/Governance → Candidate Generation → Type Routing → Promotion → Rollback
    - 数据流完整
    - 组件间接口匹配
    """
    print("\n" + "-"*70)
    print("测试 4: 完整链路验证")
    print("-"*70)
    
    try:
        from stage6_runtime_orchestrator import (
            BackboneWrapper,
            CandidateGenerator,
            TypeRouter,
            KBPromoter,
            RollbackHook,
        )
        
        config = Stage6Config(**SMOKE_TEST_CONFIG)
        
        # 验证各组件初始化
        backbone = BackboneWrapper(config)
        candidate_gen = CandidateGenerator(config)
        type_router = TypeRouter(config)
        kb_promoter = KBPromoter(config)
        rollback_hook = RollbackHook(config)
        
        # 验证数据流
        # 1. 主干推理
        input_ids = torch.randint(0, 10000, (1, 50))
        backbone_output = backbone.forward(input_ids)
        
        # 2. 候选生成
        candidates = candidate_gen.generate("测试", backbone_output, num_candidates=3)
        
        # 3. 类型分流
        if candidates:
            route = type_router.route(candidates[0])
        
        # 4. 知识库晋升
        if candidates:
            kb_result = kb_promoter.promote(candidates[0])
        
        # 5. 回滚钩子
        model = backbone.get_model()
        baseline_abilities = {'target': 0.5, 'retrieval': 0.5}
        rollback_hook.save_baseline(model, baseline_abilities)
        
        checks = {
            'backbone_init': backbone is not None,
            'candidate_gen_init': candidate_gen is not None,
            'type_router_init': type_router is not None,
            'kb_promoter_init': kb_promoter is not None,
            'rollback_hook_init': rollback_hook is not None,
            'backbone_output': backbone_output is not None,
            'candidates_generated': len(candidates) > 0,
            'routing_works': route in ['PARAM_PROMOTION', 'KB_PROMOTION', 'LONG_TERM', 'DISCARD'],
            'kb_promotion_works': kb_result.get('success', False),
            'rollback_baseline_saved': rollback_hook.baseline_state is not None,
        }
        
        all_passed = all(checks.values())
        
        details = {
            'components_initialized': 5,
            'candidates_count': len(candidates),
            'routing_result': route if candidates else 'N/A',
            'kb_entries': len(kb_promoter.kb),
        }
        
        if all_passed:
            print("✓ 完整链路验证通过")
            print(f"  - 组件初始化: {details['components_initialized']}/5")
            print(f"  - 候选生成: {details['candidates_count']} 个")
            print(f"  - 路由结果: {details['routing_result']}")
            print(f"  - 知识库条目: {details['kb_entries']}")
        else:
            print("✗ 完整链路验证失败")
            for check, passed in checks.items():
                if not passed:
                    print(f"  - 失败: {check}")
        
        return {
            'passed': all_passed,
            'details': details,
        }
        
    except Exception as e:
        print(f"✗ 完整链路验证异常: {e}")
        import traceback
        traceback.print_exc()
        return {
            'passed': False,
            'details': {'error': str(e)},
        }


def test_metrics_collection() -> Dict:
    """
    测试 5: 指标收集验证
    
    验证目标:
    - 单步指标完整
    - 累积指标正确
    - 报告导出正常
    """
    print("\n" + "-"*70)
    print("测试 5: 指标收集验证")
    print("-"*70)
    
    try:
        config = Stage6Config(**SMOKE_TEST_CONFIG)
        orchestrator = Stage6Orchestrator(config)
        
        # 使用固定的随机种子确保主干输出稳定
        import torch
        torch.manual_seed(42)
        
        # 执行两步以产生指标
        contexts = ["查询 1", "查询 2"]
        results = orchestrator.run_continuous_promotion(contexts)
        
        # 验证指标收集
        metrics_history = orchestrator.metrics_history
        
        # 验证报告生成
        report = orchestrator.get_metrics_report()
        
        # 尝试导出
        test_report_path = "eval/stage6_smoke_test_report.json"
        orchestrator.export_report(test_report_path)
        
        checks = {
            'metrics_collected': len(metrics_history) >= 2,
            'has_step_numbers': all(hasattr(m, 'step_number') for m in metrics_history),
            'has_target_improvement': all(hasattr(m, 'target_improvement') for m in metrics_history),
            'has_old_ability_drop': all(hasattr(m, 'old_ability_drop') for m in metrics_history),
            'has_writeback_change': all(hasattr(m, 'writeback_change') for m in metrics_history),
            'has_kl_weights': all(hasattr(m, 'kl_weights') for m in metrics_history),
            'report_generated': report is not None,
            'report_has_total_steps': 'total_steps' in report,
            'report_exported': Path(test_report_path).exists(),
        }
        
        all_passed = all(checks.values())
        
        details = {
            'metrics_count': len(metrics_history),
            'total_steps_in_report': report.get('total_steps', 0),
            'avg_target': f"{report.get('avg_target_improvement', 0):+.2%}",
            'max_old_drop': f"{report.get('max_old_ability_drop', 0):.2%}",
            'report_path': test_report_path,
        }
        
        if all_passed:
            print("✓ 指标收集验证通过")
            print(f"  - 收集指标数: {details['metrics_count']}")
            print(f"  - 报告步数: {details['total_steps_in_report']}")
            print(f"  - 平均目标提升: {details['avg_target']}")
            print(f"  - 最大旧能力掉落: {details['max_old_drop']}")
            print(f"  - 报告导出: {details['report_path']}")
        else:
            print("✗ 指标收集验证失败")
            for check, passed in checks.items():
                if not passed:
                    print(f"  - 失败: {check}")
        
        return {
            'passed': all_passed,
            'details': details,
        }
        
    except Exception as e:
        print(f"✗ 指标收集验证异常: {e}")
        import traceback
        traceback.print_exc()
        return {
            'passed': False,
            'details': {'error': str(e)},
        }


# ==================== 主测试函数 ====================

def run_all_smoke_tests():
    """运行所有冒烟测试"""
    print("="*70)
    print("Stage 6 Smoke Test - 全链路整合冒烟测试")
    print("="*70)
    print(f"开始时间: {datetime.now().isoformat()}")
    print(f"测试配置: 6B (base_kl=0.30/0.38, replay=0.35)")
    
    reporter = SmokeTestReporter()
    
    # 测试 1: 单步整合运行
    result1 = test_single_step_promotion()
    reporter.add_test("单步整合运行", result1['passed'], result1['details'])
    
    # 测试 2: 两步连续整合运行
    result2 = test_two_step_promotion()
    reporter.add_test("两步连续整合运行", result2['passed'], result2['details'])
    
    # 测试 3: 回滚机制验证
    result3 = test_rollback_mechanism()
    reporter.add_test("回滚机制验证", result3['passed'], result3['details'])
    
    # 测试 4: 完整链路验证
    result4 = test_integration_chain()
    reporter.add_test("完整链路验证", result4['passed'], result4['details'])
    
    # 测试 5: 指标收集验证
    result5 = test_metrics_collection()
    reporter.add_test("指标收集验证", result5['passed'], result5['details'])
    
    # 打印摘要
    reporter.print_summary()
    
    # 导出报告
    reporter.export("eval/stage6_smoke_test_summary.json")
    
    return reporter.failed == 0


if __name__ == "__main__":
    success = run_all_smoke_tests()
    sys.exit(0 if success else 1)
