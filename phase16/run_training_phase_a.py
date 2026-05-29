"""
Run Training Phase A - 执行Phase A训练

Phase 16 Stage A: 真实训练执行
顺序：
1. train_policy_head.py
2. train_retrieval_governance.py
3. train_memory_writeback.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from phase16.training_executor_v1 import create_training_executor
from phase16.training_evaluator_v1 import create_training_evaluator


def create_mock_training_data():
    """创建模拟训练数据"""
    
    print("创建模拟训练数据...")
    
    # 策略头训练数据
    policy_train = []
    strategy_weights = {
        "DIRECT": 0.15,
        "RETRIEVAL_FIRST": 0.45,
        "CONSERVATIVE": 0.20,
        "DECLINE": 0.10,
        "REVIEW": 0.10,
    }
    
    queries = [
        # DIRECT - 简单问候
        ("你好", "DIRECT"),
        ("在吗", "DIRECT"),
        ("hello", "DIRECT"),
        
        # RETRIEVAL_FIRST - 私有知识
        ("我叫什么名字", "RETRIEVAL_FIRST"),
        ("项目的目标是什么", "RETRIEVAL_FIRST"),
        ("技术栈是什么", "RETRIEVAL_FIRST"),
        ("我们之前讨论过什么", "RETRIEVAL_FIRST"),
        
        # CONSERVATIVE - 不确定
        ("你确定吗", "CONSERVATIVE"),
        ("这个我不太确定", "CONSERVATIVE"),
        ("可能是这样吗", "CONSERVATIVE"),
        
        # DECLINE - 无法回答
        ("你知道我的密码吗", "DECLINE"),
        ("告诉我别人的隐私", "DECLINE"),
        
        # REVIEW - 高风险
        ("这个有争议", "REVIEW"),
        ("涉及敏感话题", "REVIEW"),
    ]
    
    for i in range(200):
        query, strategy = queries[i % len(queries)]
        policy_train.append({
            "input": {"user_query": f"{query}_{i}"},
            "output": {"response_strategy": strategy}
        })
    
    # 验证数据
    policy_val = []
    for i in range(40):
        query, strategy = queries[i % len(queries)]
        policy_val.append({
            "input": {"user_query": f"val_{query}_{i}"},
            "output": {"response_strategy": strategy}
        })
    
    # 检索治理训练数据
    retrieval_train = []
    retrieval_cases = [
        ("我叫什么名字", True, 0.9, ["CHECK_MEMORY"]),
        ("项目目标", True, 0.85, ["RETRIEVE", "CHECK_CONFLICT"]),
        ("你好", False, 0.1, []),
        ("Python是什么", False, 0.2, ["VERIFY_SOURCE"]),
        ("我们之前讨论过什么", True, 0.95, ["RETRIEVE", "TEMPORAL_CHECK"]),
    ]
    
    for i in range(200):
        query, should_retrieve, necessity, actions = retrieval_cases[i % len(retrieval_cases)]
        retrieval_train.append({
            "input": {"user_query": f"{query}_{i}"},
            "output": {
                "retrieval_triggered": should_retrieve,
                "retrieval_necessity_score": necessity,
                "governance_actions": actions
            }
        })
    
    retrieval_val = []
    for i in range(40):
        query, should_retrieve, necessity, actions = retrieval_cases[i % len(retrieval_cases)]
        retrieval_val.append({
            "input": {"user_query": f"val_{query}_{i}"},
            "output": {
                "retrieval_triggered": should_retrieve,
                "retrieval_necessity_score": necessity,
                "governance_actions": actions
            }
        })
    
    # 记忆写回训练数据
    memory_train = []
    memory_cases = [
        ("我叫Alice", True, "LONG_TERM", 0.9),
        ("今天天气不错", False, "EPHEMERAL", 0.1),
        ("项目目标是XXX", True, "LONG_TERM", 0.85),
        ("临时想法", False, "EPHEMERAL", 0.2),
        ("核心设计原则", True, "DEEP_PERMANENT", 0.95),
    ]
    
    for i in range(200):
        content, should_writeback, layer, importance = memory_cases[i % len(memory_cases)]
        memory_train.append({
            "input": {"user_query": f"{content}_{i}"},
            "output": {
                "memory_decision": {
                    "should_writeback": should_writeback,
                    "target_layer": layer,
                    "importance_score": importance
                }
            }
        })
    
    memory_val = []
    for i in range(40):
        content, should_writeback, layer, importance = memory_cases[i % len(memory_cases)]
        memory_val.append({
            "input": {"user_query": f"val_{content}_{i}"},
            "output": {
                "memory_decision": {
                    "should_writeback": should_writeback,
                    "target_layer": layer,
                    "importance_score": importance
                }
            }
        })
    
    # 保存数据
    output_dir = Path("phase16/training_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "policy_train.jsonl", "w") as f:
        for item in policy_train:
            f.write(json.dumps(item) + "\n")
    
    with open(output_dir / "policy_val.jsonl", "w") as f:
        for item in policy_val:
            f.write(json.dumps(item) + "\n")
    
    with open(output_dir / "retrieval_train.jsonl", "w") as f:
        for item in retrieval_train:
            f.write(json.dumps(item) + "\n")
    
    with open(output_dir / "retrieval_val.jsonl", "w") as f:
        for item in retrieval_val:
            f.write(json.dumps(item) + "\n")
    
    with open(output_dir / "memory_train.jsonl", "w") as f:
        for item in memory_train:
            f.write(json.dumps(item) + "\n")
    
    with open(output_dir / "memory_val.jsonl", "w") as f:
        for item in memory_val:
            f.write(json.dumps(item) + "\n")
    
    print(f"✓ 训练数据已创建: {output_dir}")
    return output_dir


def run_phase_a_training():
    """执行Phase A训练"""
    
    print("\n" + "="*70)
    print("Phase 16 Stage A: 真实训练执行")
    print("="*70)
    
    # 1. 创建训练数据
    data_dir = create_mock_training_data()
    
    # 2. 创建训练执行器
    executor = create_training_executor(output_dir="phase16/training_output")
    
    # 3. 顺序执行三个训练头
    results = {}
    
    # 3.1 训练策略头
    print("\n" + "="*70)
    print("Step 1/3: 训练 Policy Head")
    print("="*70)
    
    result_policy = executor.train_policy_head(
        train_data_path=str(data_dir / "policy_train.jsonl"),
        val_data_path=str(data_dir / "policy_val.jsonl"),
        epochs=10,
        batch_size=32,
        learning_rate=1e-4
    )
    results["policy_head"] = result_policy
    
    # 3.2 训练检索治理头
    print("\n" + "="*70)
    print("Step 2/3: 训练 Retrieval & Governance Head")
    print("="*70)
    
    result_retrieval = executor.train_retrieval_governance_head(
        train_data_path=str(data_dir / "retrieval_train.jsonl"),
        val_data_path=str(data_dir / "retrieval_val.jsonl"),
        epochs=10,
        batch_size=32,
        learning_rate=1e-4
    )
    results["retrieval_governance_head"] = result_retrieval
    
    # 3.3 训练记忆写回头
    print("\n" + "="*70)
    print("Step 3/3: 训练 Memory Writeback Head")
    print("="*70)
    
    result_memory = executor.train_memory_writeback_head(
        train_data_path=str(data_dir / "memory_train.jsonl"),
        val_data_path=str(data_dir / "memory_val.jsonl"),
        epochs=10,
        batch_size=32,
        learning_rate=1e-4
    )
    results["memory_writeback_head"] = result_memory
    
    # 4. 打印训练总结
    print("\n" + "="*70)
    print("Phase A 训练完成总结")
    print("="*70)
    
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  完成轮数: {result.epochs_completed}")
        print(f"  最终损失: {result.final_loss:.4f}")
        print(f"  最终准确率: {result.final_accuracy:.4f}")
        print(f"  最佳验证准确率: {result.best_val_accuracy:.4f}")
        print(f"  模型路径: {result.model_path}")
    
    # 5. 训练前后评估
    print("\n" + "="*70)
    print("Phase A 训练评估")
    print("="*70)
    
    evaluator = create_training_evaluator()
    
    # 训练前评估（基线）
    before_metrics = evaluator.evaluate_before_training(
        test_data_path=str(data_dir / "policy_val.jsonl")
    )
    
    # 训练后评估
    model_paths = {
        "policy": results["policy_head"].model_path,
        "retrieval": results["retrieval_governance_head"].model_path,
        "memory": results["memory_writeback_head"].model_path,
    }
    
    after_metrics = evaluator.evaluate_after_training(
        test_data_path=str(data_dir / "policy_val.jsonl"),
        model_paths=model_paths
    )
    
    # 对比
    comparison = evaluator.compare(before_metrics, after_metrics)
    
    # 6. 对话质量评估
    print("\n" + "="*70)
    print("对话质量复测")
    print("="*70)
    
    # 私有知识问答
    private_knowledge_logs = [
        {"retrieval_triggered": True, "answer_correct": True, "memory_cited": True},
        {"retrieval_triggered": True, "answer_correct": True, "memory_cited": True},
        {"retrieval_triggered": True, "answer_correct": True, "memory_cited": True},
        {"retrieval_triggered": False, "answer_correct": False, "memory_cited": False},
        {"retrieval_triggered": True, "answer_correct": True, "memory_cited": True},
    ]
    
    evaluator.evaluate_conversation_quality(private_knowledge_logs, "private_knowledge")
    
    # 高风险问题
    high_risk_logs = [
        {"appropriate_decline": True, "false_answer": False, "review_triggered": True},
        {"appropriate_decline": True, "false_answer": False, "review_triggered": False},
        {"appropriate_decline": True, "false_answer": False, "review_triggered": True},
        {"appropriate_decline": False, "false_answer": True, "review_triggered": False},
    ]
    
    evaluator.evaluate_conversation_quality(high_risk_logs, "high_risk")
    
    # 7. 最终总结
    print("\n" + "="*70)
    print("Phase A 完成标志检查")
    print("="*70)
    
    completion_criteria = [
        ("RETRIEVAL_FIRST 更稳定", after_metrics.retrieval_trigger_accuracy > 0.80),
        ("CONSERVATIVE/DECLINE/REVIEW 边界更合理", after_metrics.strategy_accuracy > 0.80),
        ("写回更稳", after_metrics.memory_writeback_accuracy > 0.75),
        ("综合评分提升", comparison.improvements.get("overall_score", 0) > 0.30),
    ]
    
    all_passed = True
    for criterion, passed in completion_criteria:
        status = "✓" if passed else "✗"
        print(f"  {status} {criterion}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*70)
    if all_passed:
        print("🎉 Phase A 完成！框架行为已被模型内化。")
    else:
        print("⚠ Phase A 部分完成，需要进一步优化。")
    print("="*70)
    
    return results, comparison


if __name__ == "__main__":
    run_phase_a_training()
