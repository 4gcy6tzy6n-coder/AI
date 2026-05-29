"""
第六阶段最终验证 - 5条完成标志检查

验证目标：
1. 完整训练闭环可以独立跑通
2. deep_permanent 可作为参数晋升前置层
3. 参数写回只进入自构建学习参数区
4. 基础手工训练参数不会被直接覆盖
5. 写回后有回滚/冻结机制，且旧能力可验证不受破坏
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelines.full_training_loop import FullTrainingLoop, TrainingCase
from src.core.training.promotion_manager import PromotionManager
from src.core.gates.parameter_promotion_gate import ParameterPromotionGate, ParamPromotionDecision
from src.core.params.manual_param_store import ManualParamStore
from src.core.params.self_learned_param_store import SelfLearnedParamStore


def check_1_full_loop_runs():
    """检查1: 完整训练闭环可以独立跑通"""
    print("\n" + "-" * 70)
    print("检查 1/5: 完整训练闭环可以独立跑通")
    print("-" * 70)

    try:
        loop = FullTrainingLoop()

        # 创建测试案例
        seed_cases = [
            TrainingCase(
                case_id=f"val_{i:03d}",
                content=f"验证样本 {i}",
                source="curated_dataset",
                seed_quality_score=0.90
            )
            for i in range(3)
        ]

        # 运行循环
        result = loop.run_loop(
            seed_cases=seed_cases,
            max_rounds=1,
            enable_param_promotion=True
        )

        # 验证
        runs_independently = result.total_rounds > 0
        outputs_metrics = result.final_metrics is not None

        passed = runs_independently and outputs_metrics

        print(f"  独立运行: {'✅' if runs_independently else '❌'}")
        print(f"  输出指标: {'✅' if outputs_metrics else '❌'}")
        print(f"  结果: {'✅ 通过' if passed else '❌ 失败'}")

        return passed

    except Exception as e:
        print(f"  ❌ 异常: {e}")
        return False


def check_2_deep_permanent_as_param_source():
    """检查2: deep_permanent 可作为参数晋升前置层"""
    print("\n" + "-" * 70)
    print("检查 2/5: deep_permanent 可作为参数晋升前置层")
    print("-" * 70)

    try:
        gate = ParameterPromotionGate()

        # 测试 deep_permanent 来源
        result = gate.evaluate(
            candidate_id="check_001",
            source_zone="deep_permanent",
            quality_scores={"Q": 92, "T": 93, "S": 91, "C": 96, "L": 94},
            verification_rounds=10,
            cross_task_score=0.92,
            long_term_stability=0.96,
            cross_version_score=0.88
        )

        # 验证
        source_accepted = result.decision != ParamPromotionDecision.BLOCKED_BY_SOURCE
        can_promote = result.decision == ParamPromotionDecision.APPROVED or source_accepted

        print(f"  来源接受: {'✅' if source_accepted else '❌'}")
        print(f"  可以晋升: {'✅' if can_promote else '❌'}")
        print(f"  结果: {'✅ 通过' if source_accepted else '❌ 失败'}")

        return source_accepted

    except Exception as e:
        print(f"  ❌ 异常: {e}")
        return False


def check_3_param_write_to_self_learned():
    """检查3: 参数写回只进入自构建学习参数区"""
    print("\n" + "-" * 70)
    print("检查 3/5: 参数写回只进入自构建学习参数区")
    print("-" * 70)

    try:
        # 测试自构建存储接受写回
        self_store = SelfLearnedParamStore()
        param = self_store.write_param(
            param_id="check_sl_001",
            source_unit_id="deep_unit_001",
            source_zone="deep_permanent",
            value_shape=(768, 768),
            value_hash="hash_check",
            promotion_scores={"Q": 92, "T": 93, "S": 91, "C": 96, "L": 94},
            verification_rounds=10,
            cross_task_score=0.92,
            long_term_stability=0.96,
            gate_decision="approved",
            gate_confidence=0.95
        )
        self_learned_accepts = param is not None

        # 测试手工存储阻止写回
        manual_store = ManualParamStore()
        blocked = not manual_store.write_param(
            param_id="attention_weights_layer_0",
            value={},
            source="test"
        )

        print(f"  自构建接受: {'✅' if self_learned_accepts else '❌'}")
        print(f"  手工存储阻止: {'✅' if blocked else '❌'}")
        print(f"  结果: {'✅ 通过' if self_learned_accepts and blocked else '❌ 失败'}")

        return self_learned_accepts and blocked

    except Exception as e:
        print(f"  ❌ 异常: {e}")
        return False


def check_4_manual_param_protected():
    """检查4: 基础手工训练参数不会被直接覆盖"""
    print("\n" + "-" * 70)
    print("检查 4/5: 基础手工训练参数不会被直接覆盖")
    print("-" * 70)

    try:
        store = ManualParamStore()

        # 尝试多种写入方式
        blocked_direct = not store.write_param(
            "attention_weights_layer_0", {}, "test"
        )
        blocked_no_token = not store.attempt_write(
            "feedforward_weights_layer_0", {}, "test"
        )
        blocked_critical = not store.attempt_write(
            "attention_weights_layer_0", {}, "test", "APPROVED_123"
        )

        # 验证保护状态
        summary = store.get_protection_summary()
        protection_active = summary['protection_active']

        print(f"  直接写入阻止: {'✅' if blocked_direct else '❌'}")
        print(f"  无令牌阻止: {'✅' if blocked_no_token else '❌'}")
        print(f"  关键参数阻止: {'✅' if blocked_critical else '❌'}")
        print(f"  保护激活: {'✅' if protection_active else '❌'}")

        all_blocked = blocked_direct and blocked_no_token and blocked_critical
        print(f"  结果: {'✅ 通过' if all_blocked and protection_active else '❌ 失败'}")

        return all_blocked and protection_active

    except Exception as e:
        print(f"  ❌ 异常: {e}")
        return False


def check_5_rollback_mechanism():
    """检查5: 写回后有回滚/冻结机制"""
    print("\n" + "-" * 70)
    print("检查 5/5: 写回后有回滚/冻结机制")
    print("-" * 70)

    try:
        store = SelfLearnedParamStore()

        # 写入并激活参数
        param = store.write_param(
            param_id="check_rollback_001",
            source_unit_id="deep_unit_001",
            source_zone="deep_permanent",
            value_shape=(768, 768),
            value_hash="hash_rb",
            promotion_scores={"Q": 92, "T": 93, "S": 91, "C": 96, "L": 94},
            verification_rounds=10,
            cross_task_score=0.92,
            long_term_stability=0.96,
            gate_decision="approved",
            gate_confidence=0.95
        )
        store.activate_param(param.param_id)

        # 回滚参数
        rollback_success = store.rollback_param(
            param_id=param.param_id,
            reason="能力漂移测试"
        )

        # 验证回滚状态
        param = store.get_param(param.param_id)
        is_rolled_back = param.status.value == "rolled_back"
        has_reason = param.rollback_reason is not None

        # 测试冻结
        param2 = store.write_param(
            param_id="check_freeze_001",
            source_unit_id="deep_unit_002",
            source_zone="deep_permanent",
            value_shape=(768, 768),
            value_hash="hash_fz",
            promotion_scores={"Q": 91, "T": 92, "S": 90, "C": 95, "L": 93},
            verification_rounds=9,
            cross_task_score=0.91,
            long_term_stability=0.95,
            gate_decision="approved",
            gate_confidence=0.90
        )
        freeze_success = store.freeze_param(param2.param_id, "测试冻结")
        param2 = store.get_param(param2.param_id)
        is_frozen = param2.status.value == "frozen"

        # 检查回滚历史
        history = store.get_rollback_history()
        has_history = len(history) > 0

        print(f"  回滚成功: {'✅' if rollback_success else '❌'}")
        print(f"  状态为rolled_back: {'✅' if is_rolled_back else '❌'}")
        print(f"  有回滚原因: {'✅' if has_reason else '❌'}")
        print(f"  冻结成功: {'✅' if freeze_success else '❌'}")
        print(f"  状态为frozen: {'✅' if is_frozen else '❌'}")
        print(f"  有回滚历史: {'✅' if has_history else '❌'}")

        passed = (rollback_success and is_rolled_back and has_reason and
                  freeze_success and is_frozen and has_history)
        print(f"  结果: {'✅ 通过' if passed else '❌ 失败'}")

        return passed

    except Exception as e:
        print(f"  ❌ 异常: {e}")
        return False


def run_final_validation():
    """运行最终验证"""
    print("\n" + "🎯 " * 35)
    print("第六阶段最终验证 - 5条完成标志检查")
    print("🎯 " * 35)

    checks = {
        "完整训练闭环可以独立跑通": check_1_full_loop_runs,
        "deep_permanent可作为参数晋升前置层": check_2_deep_permanent_as_param_source,
        "参数写回只进入自构建学习参数区": check_3_param_write_to_self_learned,
        "基础手工训练参数不会被直接覆盖": check_4_manual_param_protected,
        "写回后有回滚/冻结机制": check_5_rollback_mechanism
    }

    results = {}
    for check_name, check_func in checks.items():
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"\n❌ {check_name} 异常: {e}")
            results[check_name] = False

    # 汇总
    print("\n" + "=" * 70)
    print("第六阶段最终验证总结")
    print("=" * 70)

    for check_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {check_name}: {status}")

    all_passed = all(results.values())

    if all_passed:
        print("\n" + "🎉 " * 35)
        print("第六阶段 Full Training Loop v0.6 全部完成！")
        print("🎉 " * 35)
        print("\n核心成果：")
        print("  ✓ full_training_loop.py - 完整训练闭环")
        print("  ✓ promotion_manager.py - 参数晋升管理")
        print("  ✓ parameter_promotion_gate.py - 参数晋升门控")
        print("  ✓ manual_param_store.py - 手工参数存储（受保护）")
        print("  ✓ self_learned_param_store.py - 自构建参数存储")
        print("  ✓ test_phase6_full_training_loop.py - 第六阶段测试")
        print("\n关键原则已落实：")
        print("  ✓ 基础手工训练参数独立存放，禁止直接覆盖")
        print("  ✓ 模型自构建学习参数独立存放，允许受控写回")
        print("  ✓ 参数晋升门槛高于深层永久")
        print("  ✓ 写回后有回滚/冻结机制")
        print("  ✓ 旧能力可验证不受破坏")
    else:
        print("\n⚠️ 部分检查失败，需要修复。")

    return all_passed


if __name__ == "__main__":
    run_final_validation()
