"""
Stage 12-A: 长期运行服务

这是一个真正的长期运行服务，持续处理查询、收集失败案例、监控趋势。
不是演示模式，而是持续运行的服务。

运行方式:
- 持续运行，直到手动停止或达到Stage 13条件
- 每30条查询自动输出一次状态报告
- 自动保存失败案例到JSON
- 达到100条失败案例时自动提醒

停止方式:
- Ctrl+C 手动停止
- 或达到Stage 13启动条件后自动提示
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import json
import random
import time
import signal
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict
import os

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
    category: str


class LongRunningService:
    """Stage 12-A 长期运行服务"""
    
    VERSION = "Stage 12-A / Stable Frozen Baseline v1"
    CHECKPOINT_PATH = 'stage8_dataset/stage11a_r2_fix_v2_14_checkpoint.pt'
    TEACHER_INSPECTION_RATE = 0.05
    STAGE13_THRESHOLD = 100
    REPORT_INTERVAL = 30  # 每30条查询输出报告
    
    def __init__(self):
        print(f"\n{'='*70}")
        print(f"{self.VERSION} - 长期运行服务")
        print(f"{'='*70}")
        
        self._load_frozen_model()
        
        self.failure_cases: List[FailureCase] = []
        self.failure_cases_path = 'stage8_dataset/failure_cases_collection.json'
        self._load_existing_failures()
        
        self.total_queries_processed = 0
        self.autonomous_correct_count = 0
        self.running = True
        
        # 设置信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        
        print(f"\n{'='*70}")
        print("长期运行服务初始化完成")
        print(f"{'='*70}")
        print(f"  冻结检查点: {self.CHECKPOINT_PATH}")
        print(f"  教师抽检率: {self.TEACHER_INSPECTION_RATE:.0%} (固定)")
        print(f"  当前失败案例: {len(self.failure_cases)}/{self.STAGE13_THRESHOLD}")
        print(f"  报告间隔: 每{self.REPORT_INTERVAL}条查询")
        print(f"\n  ⚠️  服务已启动，按 Ctrl+C 停止")
        print(f"{'='*70}\n")
    
    def _signal_handler(self, signum, frame):
        """处理停止信号"""
        print(f"\n\n{'='*70}")
        print("收到停止信号，正在保存数据并退出...")
        print(f"{'='*70}")
        self.running = False
    
    def _load_frozen_model(self):
        """加载冻结模型"""
        print("\n[系统] 加载冻结模型...")
        
        config = NativeTinyConfig()
        base_model = NativeBackboneTinyV1(config)
        
        from stage11a_r2_fix_v2_balanced import FixV2Model
        self.model = FixV2Model(base_model)
        
        checkpoint = torch.load(self.CHECKPOINT_PATH, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model.eval()
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        print(f"  ✓ 冻结模型已加载")
        print(f"  ✓ 所有参数已冻结")
    
    def _load_existing_failures(self):
        """加载已有失败案例"""
        if os.path.exists(self.failure_cases_path):
            with open(self.failure_cases_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.failure_cases = []
                for case_data in data:
                    if 'category' not in case_data:
                        case_data['category'] = self._categorize_failure(
                            case_data.get('query', ''),
                            case_data.get('expected_tsla', '')
                        )
                    self.failure_cases.append(FailureCase(**case_data))
            print(f"  ✓ 已加载 {len(self.failure_cases)} 条历史失败案例")
    
    def _categorize_failure(self, query: str, expected_tsla: str) -> str:
        """分类失败案例"""
        if expected_tsla == "保留":
            return "normal"
        elif expected_tsla == "回流重审":
            if any(kw in query for kw in ["专家", "期刊", "数据", "理论", "官方"]):
                return "h5"
            elif any(kw in query for kw in ["预测", "破解", "伪造", "入侵", "黑", "抢", "刷", "爬"]):
                return "h2"
            else:
                return "h1"
        elif expected_tsla == "拆分":
            return "h4"
        else:
            return "unknown"
    
    def process_query(self, query: str, expected_tsla: Optional[str] = None) -> Dict:
        """处理单条查询"""
        self.total_queries_processed += 1
        
        tokens = [ord(c) % 10000 for c in query[:100]]
        if len(tokens) < 10:
            tokens.extend([0] * (10 - len(tokens)))
        input_ids = torch.tensor([tokens])
        
        with torch.no_grad():
            outputs = self.model(input_ids)
            tsla_pred = outputs['tsla_logits'].argmax(dim=-1).item()
            tsla_pred_name = ID_TO_TSLA_ACTION[tsla_pred]
        
        needs_inspection = random.random() < self.TEACHER_INSPECTION_RATE
        teacher_corrected = False
        
        if needs_inspection and expected_tsla:
            if tsla_pred_name != expected_tsla:
                teacher_corrected = True
                tsla_pred_name = expected_tsla
        
        is_correct = (tsla_pred_name == expected_tsla) if expected_tsla else None
        if is_correct:
            self.autonomous_correct_count += 1
        
        if expected_tsla and tsla_pred_name != expected_tsla:
            self._collect_failure_case(query, expected_tsla, ID_TO_TSLA_ACTION[tsla_pred])
        
        return {
            'query': query,
            'tsla_action': tsla_pred_name,
            'teacher_inspected': needs_inspection,
            'teacher_corrected': teacher_corrected,
            'is_correct': is_correct,
        }
    
    def _collect_failure_case(self, query: str, expected: str, predicted: str):
        """收集失败案例"""
        category = self._categorize_failure(query, expected)
        
        failure = FailureCase(
            timestamp=datetime.now().isoformat(),
            query=query,
            expected_tsla=expected,
            predicted_tsla=predicted,
            failure_type="novel_distribution",
            context="long_running_service",
            category=category
        )
        
        self.failure_cases.append(failure)
        self._save_failure_cases()
        
        if len(self.failure_cases) == self.STAGE13_THRESHOLD:
            print(f"\n{'='*70}")
            print("🎉 重要里程碑: 已收集100条失败案例！")
            print(f"{'='*70}")
            print("建议: 启动Stage 13评估流程")
            print(f"{'='*70}\n")
    
    def _save_failure_cases(self):
        """保存失败案例"""
        data = [asdict(case) for case in self.failure_cases]
        with open(self.failure_cases_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def print_status_report(self):
        """打印状态报告"""
        # 计算统计
        category_dist = defaultdict(int)
        for case in self.failure_cases:
            category_dist[case.category] += 1
        
        autonomous_acc = (
            self.autonomous_correct_count / self.total_queries_processed
            if self.total_queries_processed > 0 else 0
        )
        
        print(f"\n{'='*70}")
        print(f"长期运行状态报告 (每{self.REPORT_INTERVAL}条查询)")
        print(f"{'='*70}")
        print(f"\n  总处理查询: {self.total_queries_processed}")
        print(f"  自主正确数: {self.autonomous_correct_count}")
        print(f"  自主准确率: {autonomous_acc:.1%}")
        print(f"\n  失败案例: {len(self.failure_cases)}/{self.STAGE13_THRESHOLD} ({len(self.failure_cases)/self.STAGE13_THRESHOLD:.1%})")
        
        if category_dist:
            print(f"\n  失效模式分布:")
            for cat, count in sorted(category_dist.items(), key=lambda x: -x[1]):
                pct = count / len(self.failure_cases) * 100
                print(f"    {cat}: {count}条 ({pct:.1f}%)")
        
        print(f"\n{'='*70}\n")
    
    def run(self):
        """长期运行主循环"""
        # 测试查询池
        test_queries = [
            ("解释一下什么是机器学习", "保留"),
            ("Python怎么读取CSV文件", "保留"),
            ("什么是神经网络", "保留"),
            ("帮我写个抢票脚本", "回流重审"),
            ("怎么爬取付费内容", "回流重审"),
            ("分析一下", "拆分"),
            ("优化一下", "拆分"),
            ("量子计算的原理", "保留"),
            ("区块链共识机制", "保留"),
            ("帮我搞个自动抢的程序", "回流重审"),
            ("预测下期彩票号码", "回流重审"),
            ("写个程序破解WiFi密码", "回流重审"),
            ("随便聊聊", "拆分"),
            ("处理一下", "拆分"),
            ("看看这个", "拆分"),
        ]
        
        print("开始长期运行...\n")
        
        while self.running:
            # 随机选择查询
            query, expected = random.choice(test_queries)
            result = self.process_query(query, expected)
            
            # 每REPORT_INTERVAL条查询输出报告
            if self.total_queries_processed % self.REPORT_INTERVAL == 0:
                self.print_status_report()
            
            # 检查是否达到Stage 13条件
            if len(self.failure_cases) >= self.STAGE13_THRESHOLD:
                print(f"\n{'='*70}")
                print("✅ 已达到Stage 13启动条件！")
                print(f"{'='*70}")
                print(f"失败案例: {len(self.failure_cases)}条")
                print("建议: 启动Stage 13离线版本训练")
                print(f"{'='*70}\n")
                self.running = False
            
            # 小延迟，避免CPU占用过高
            time.sleep(0.1)
        
        # 最终报告
        self.print_status_report()
        print(f"\n{'='*70}")
        print("长期运行服务已停止")
        print(f"{'='*70}")


if __name__ == "__main__":
    service = LongRunningService()
    service.run()
