"""
Run Calibration - 执行定标实验

用法：
    python scripts/run_calibration.py --mode grid_search
    python scripts/run_calibration.py --mode sensitivity --param stable_min_q
    python scripts/run_calibration.py --mode single --config config.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipelines.calibration_pipeline import CalibrationPipeline, ThresholdConfig


def print_banner():
    """打印横幅"""
    print("\n" + "=" * 70)
    print(" " * 20 + "TSLA 阈值定标实验")
    print("=" * 70)
    print("\n目标：从示例参数推进到第一版实验参数")
    print("方法：网格搜索 + 敏感性分析 + 假阳性/假阴性统计\n")


def run_grid_search(args):
    """运行网格搜索"""
    print("\n" + "-" * 70)
    print("模式：网格搜索")
    print("-" * 70)

    pipeline = CalibrationPipeline()
    pipeline.load_cases()

    # 定义参数网格
    param_grid = {
        "stable_min_q": [68, 70, 72],
        "stable_min_s": [63, 65, 67],
        "max_q_std": [8, 10, 12],
        "max_conflict_rate": [0.15, 0.2, 0.25],
        "peak_drop_q": [12, 15, 18],
        "peak_drop_s": [12, 15, 18]
    }

    print(f"\n参数网格：")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")

    total_combinations = 1
    for values in param_grid.values():
        total_combinations *= len(values)

    print(f"\n总组合数: {total_combinations}")
    print("开始搜索...")

    results = pipeline.grid_search(param_grid)

    # 显示前10个最优结果
    print("\n" + "-" * 70)
    print("Top 10 最优配置（按F1分数排序）：")
    print("-" * 70)

    print(f"\n{'Rank':<6}{'Q_min':<8}{'S_min':<8}{'Q_std':<8}{'Conflict':<10}{'Accuracy':<10}{'F1':<8}")
    print("-" * 70)

    for i, result in enumerate(results[:10], 1):
        cfg = result.config
        print(f"{i:<6}{cfg.stable_min_q:<8.0f}{cfg.stable_min_s:<8.0f}{cfg.max_q_std:<8.0f}"
              f"{cfg.max_conflict_rate:<10.2f}{result.accuracy:<10.2%}{result.f1_score:<8.3f}")

    # 显示最优配置的详细指标
    best = results[0]
    print("\n" + "-" * 70)
    print("最优配置详细指标：")
    print("-" * 70)
    print(f"  准确率 (Accuracy):  {best.accuracy:.2%}")
    print(f"  精确率 (Precision): {best.precision:.2%}")
    print(f"  召回率 (Recall):    {best.recall:.2%}")
    print(f"  F1分数:             {best.f1_score:.3f}")
    print(f"  假阳性:             {best.false_positives}")
    print(f"  假阴性:             {best.false_negatives}")

    # 导出结果
    output_path = Path(__file__).parent.parent / "outputs" / "calibration_results.json"
    output_path.parent.mkdir(exist_ok=True)
    pipeline.export_results(str(output_path))
    print(f"\n结果已导出到: {output_path}")

    return results


def run_sensitivity_analysis(args):
    """运行敏感性分析"""
    print("\n" + "-" * 70)
    print(f"模式：敏感性分析 - 参数: {args.param}")
    print("-" * 70)

    pipeline = CalibrationPipeline()
    pipeline.load_cases()

    # 定义参数范围
    param_ranges = {
        "stable_min_q": [65, 68, 70, 72, 75],
        "stable_min_s": [60, 63, 65, 67, 70],
        "max_q_std": [5, 8, 10, 12, 15],
        "max_conflict_rate": [0.1, 0.15, 0.2, 0.25, 0.3],
        "peak_drop_q": [10, 12, 15, 18, 20],
        "peak_drop_s": [10, 12, 15, 18, 20]
    }

    if args.param not in param_ranges:
        print(f"错误：未知参数 {args.param}")
        print(f"可用参数: {list(param_ranges.keys())}")
        return

    param_range = param_ranges[args.param]
    results = pipeline.analyze_sensitivity(args.param, param_range)

    print(f"\n参数 {args.param} 的敏感性分析：")
    print(f"\n{'Value':<10}{'Accuracy':<12}{'Precision':<12}{'Recall':<10}{'F1':<10}{'FP':<6}{'FN':<6}")
    print("-" * 70)

    for r in results:
        print(f"{r['param_value']:<10}{r['accuracy']:<12.2%}{r['precision']:<12.2%}"
              f"{r['recall']:<10.2%}{r['f1_score']:<10.3f}{r['false_positives']:<6}{r['false_negatives']:<6}")

    # 找出最优值
    best = max(results, key=lambda x: x['f1_score'])
    print(f"\n最优值: {best['param_value']} (F1={best['f1_score']:.3f})")

    return results


def run_single_config(args):
    """运行单组配置"""
    print("\n" + "-" * 70)
    print("模式：单配置测试")
    print("-" * 70)

    # 加载配置
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = ThresholdConfig(**config_dict)
    else:
        config = ThresholdConfig()

    print("\n当前配置：")
    print(f"  stable_min_q: {config.stable_min_q}")
    print(f"  stable_min_s: {config.stable_min_s}")
    print(f"  max_q_std: {config.max_q_std}")
    print(f"  max_conflict_rate: {config.max_conflict_rate}")
    print(f"  peak_drop_q: {config.peak_drop_q}")
    print(f"  peak_drop_s: {config.peak_drop_s}")

    pipeline = CalibrationPipeline()
    pipeline.load_cases()

    result = pipeline.run_single_config(config)

    print("\n测试结果：")
    print(f"  总用例数: {result.total_cases}")
    print(f"  正确预测: {result.correct_predictions}")
    print(f"  假阳性: {result.false_positives}")
    print(f"  假阴性: {result.false_negatives}")
    print(f"  准确率: {result.accuracy:.2%}")
    print(f"  精确率: {result.precision:.2%}")
    print(f"  召回率: {result.recall:.2%}")
    print(f"  F1分数: {result.f1_score:.3f}")

    # 显示每个用例的结果
    print("\n详细结果：")
    print(f"{'Case ID':<25}{'Category':<20}{'Expected':<12}{'Actual':<12}{'Status':<8}")
    print("-" * 80)

    for case in result.case_results:
        expected = "Promote" if case["expected_promotion"] else "Block"
        actual = "Promote" if case["promotion_decision"] == "promote_to_normal" else "Block"
        status = "✅" if case["correct"] else "❌"
        category = case["category"][:18]
        print(f"{case['case_id']:<25}{category:<20}{expected:<12}{actual:<12}{status:<8}")

    return result


def main():
    parser = argparse.ArgumentParser(description="TSLA 阈值定标实验")
    parser.add_argument("--mode", choices=["grid_search", "sensitivity", "single"],
                       default="grid_search", help="运行模式")
    parser.add_argument("--param", type=str, help="敏感性分析参数名")
    parser.add_argument("--config", type=str, help="单配置模式配置文件路径")

    args = parser.parse_args()

    print_banner()

    if args.mode == "grid_search":
        run_grid_search(args)
    elif args.mode == "sensitivity":
        if not args.param:
            print("错误：敏感性分析模式需要指定 --param 参数")
            return
        run_sensitivity_analysis(args)
    elif args.mode == "single":
        run_single_config(args)

    print("\n" + "=" * 70)
    print("定标实验完成")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
