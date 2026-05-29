"""
项目完成触发器 - 自动检测项目完成并启动自学习

功能:
1. 监控项目状态文件
2. 检测训练完成信号
3. 自动启动自学习系统
4. 持续学习直到达到目标
"""

import json
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Callable

from auto_self_learning import AutoSelfLearning


class ProjectCompletionTrigger:
    """项目完成触发器"""
    
    def __init__(
        self,
        status_file: str = "PROJECT_STATUS.md",
        checkpoint_dir: str = "stage8_dataset",
    ):
        self.status_file = Path(status_file)
        self.checkpoint_dir = Path(checkpoint_dir)
        
        # 自学习系统
        self.learner: Optional[AutoSelfLearning] = None
        
        # 监控状态
        self.monitoring = False
        self.monitor_thread = None
        
        # 回调函数
        self.on_completion: Optional[Callable] = None
        self.on_learning_start: Optional[Callable] = None
        self.on_learning_progress: Optional[Callable] = None
        
        # 配置
        self.check_interval = 30  # 检查间隔(秒)
        self.learning_duration = 600  # 学习时长(秒)
    
    def start_monitoring(self):
        """开始监控项目状态"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 项目完成监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 项目完成监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            if self._check_completion():
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🎉 项目完成信号检测到!")
                
                # 触发回调
                if self.on_completion:
                    self.on_completion()
                
                # 启动自学习
                self._start_auto_learning()
                
                # 停止监控(或继续监控下一个周期)
                break
            
            time.sleep(self.check_interval)
    
    def _check_completion(self) -> bool:
        """检查项目是否完成"""
        # 检查1: 状态文件中的完成标记
        if self.status_file.exists():
            content = self.status_file.read_text(encoding='utf-8')
            
            # 检查完成标记
            completion_markers = [
                "Stage 10-3: 完成",
                "product_baseline_v2",
                "教师模式增强完成",
                "✅ 完成",
                "COMPLETED",
            ]
            
            for marker in completion_markers:
                if marker in content:
                    return True
        
        # 检查2: 检查点文件存在
        checkpoint_files = [
            "teacher_mode_checkpoint_v2_large.pt",
            "teacher_mode_checkpoint_v2.pt",
        ]
        
        for ckpt in checkpoint_files:
            if (self.checkpoint_dir / ckpt).exists():
                return True
        
        return False
    
    def _start_auto_learning(self):
        """启动自动自学习"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 启动自动自学习系统...")
        
        # 创建学习系统
        self.learner = AutoSelfLearning()
        
        # 触发回调
        if self.on_learning_start:
            self.on_learning_start()
        
        # 启动学习
        self.learner.start()
        
        # 监控学习进度
        start_time = time.time()
        
        try:
            while time.time() - start_time < self.learning_duration:
                time.sleep(10)
                
                # 获取进度
                metrics = self.learner.get_metrics()
                
                # 触发进度回调
                if self.on_learning_progress:
                    self.on_learning_progress(metrics)
                
                # 打印进度
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"学习进度: {metrics['total_episodes']} 片段 | "
                      f"知识库: {metrics['kb_size_current']} 条")
            
            # 停止学习
            self.learner.stop()
            
            # 导出结果
            self._export_results()
            
        except KeyboardInterrupt:
            print("\n用户中断学习")
            self.learner.stop()
    
    def _export_results(self):
        """导出学习结果"""
        if not self.learner:
            return
        
        # 导出知识
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        knowledge_file = self.checkpoint_dir / f"auto_learned_knowledge_{timestamp}.json"
        self.learner.export_knowledge(str(knowledge_file))
        
        # 导出指标
        metrics = self.learner.get_metrics()
        metrics_file = self.checkpoint_dir / f"auto_learning_metrics_{timestamp}.json"
        
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ✅ 学习结果已导出:")
        print(f"  知识库: {knowledge_file}")
        print(f"  指标: {metrics_file}")
        
        # 打印总结
        print(f"\n📊 自学习总结:")
        print(f"  总学习片段: {metrics['total_episodes']}")
        print(f"  成功学习: {metrics['successful_learnings']}")
        print(f"  晋升长期记忆: {metrics['promoted_to_long_term']}")
        print(f"  最终教师依赖度: {metrics['teacher_dependency_ratio']:.2%}")
        print(f"  自主检索率: {metrics['self_retrieval_rate']:.2%}")
        print(f"  知识库增长: {metrics['kb_size_start']} -> {metrics['kb_size_current']}")


def manual_trigger_learning(duration: int = 300):
    """手动触发自学习"""
    print("="*70)
    print("🚀 手动触发自学习")
    print("="*70)
    
    trigger = ProjectCompletionTrigger()
    trigger.learning_duration = duration
    
    # 直接启动学习
    trigger._start_auto_learning()


def integrated_mode():
    """
    集成模式 - 与主项目集成
    在训练脚本最后调用此函数启动自学习
    """
    print("\n" + "="*70)
    print("🔄 进入自动自学习阶段")
    print("="*70)
    
    trigger = ProjectCompletionTrigger()
    
    # 设置回调
    def on_completion():
        print("[Callback] 项目完成确认，准备启动自学习...")
    
    def on_learning_start():
        print("[Callback] 自学习已开始")
    
    def on_learning_progress(metrics):
        # 可以在这里发送通知、更新UI等
        pass
    
    trigger.on_completion = on_completion
    trigger.on_learning_start = on_learning_start
    trigger.on_learning_progress = on_learning_progress
    
    # 开始监控
    trigger.start_monitoring()
    
    try:
        # 保持运行
        while trigger.monitoring:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n用户中断")
        trigger.stop_monitoring()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "manual":
            # 手动模式
            duration = int(sys.argv[2]) if len(sys.argv) > 2 else 300
            manual_trigger_learning(duration)
        elif sys.argv[1] == "integrated":
            # 集成模式
            integrated_mode()
        else:
            print("用法: python project_completion_trigger.py [manual|integrated] [duration]")
    else:
        # 默认手动触发
        manual_trigger_learning(300)
