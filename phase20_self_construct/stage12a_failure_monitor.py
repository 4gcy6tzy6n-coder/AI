"""
Stage 12-A: 失败案例监控与早期预警系统

阶段定义:
虽然冻结策略正确，但线上早期数据显示H1失败占主导(2/3)，
这不是随机噪声，值得提前审查。

监控规则:
1. 持续收集失败案例
2. 每20-30条查询进行一次失效模式分析
3. 如某类失败持续主导，提前开小型失效审查
4. 小型失效审查: 只做归因和分类整理，不做在线训练

预警阈值:
- 总查询数达到30条时，触发第一次失效模式分析
- 如某类失败占比>50%，标记为"需关注模式"
- 如某类失败占比>70%，触发小型失效审查
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from typing import Dict, List
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime


@dataclass
class FailurePattern:
    """失效模式"""
    category: str
    count: int
    percentage: float
    examples: List[str]
    risk_level: str  # low / medium / high


class FailureMonitor:
    """失败案例监控器"""
    
    def __init__(self, failure_cases_path: str = 'stage8_dataset/failure_cases_collection.json'):
        self.failure_cases_path = failure_cases_path
        self.analysis_threshold = 30  # 每30条查询分析一次
        self.high_risk_threshold = 0.70  # 70%为高风险
        self.medium_risk_threshold = 0.50  # 50%为中风险
        
    def load_failure_cases(self) -> List[Dict]:
        """加载失败案例"""
        try:
            with open(self.failure_cases_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
    
    def analyze_failure_patterns(self, total_queries: int) -> Dict:
        """分析失效模式"""
        failures = self.load_failure_cases()
        
        if not failures:
            return {
                'status': 'no_failures',
                'message': '暂无失败案例',
                'patterns': [],
            }
        
        # 按类别统计
        category_counts = defaultdict(lambda: {'count': 0, 'examples': []})
        for case in failures:
            cat = case.get('category', 'unknown')
            category_counts[cat]['count'] += 1
            if len(category_counts[cat]['examples']) < 3:  # 保留3个示例
                category_counts[cat]['examples'].append(case.get('query', '')[:50])
        
        # 计算模式和风险等级
        patterns = []
        for cat, data in category_counts.items():
            count = data['count']
            percentage = count / len(failures) if failures else 0
            
            if percentage >= self.high_risk_threshold:
                risk = 'high'
            elif percentage >= self.medium_risk_threshold:
                risk = 'medium'
            else:
                risk = 'low'
            
            patterns.append(FailurePattern(
                category=cat,
                count=count,
                percentage=percentage,
                examples=data['examples'],
                risk_level=risk
            ))
        
        # 按占比排序
        patterns.sort(key=lambda x: x.percentage, reverse=True)
        
        # 判断是否需要审查
        needs_review = any(p.risk_level == 'high' for p in patterns)
        needs_attention = any(p.risk_level == 'medium' for p in patterns)
        
        return {
            'status': 'analysis_complete',
            'total_queries': total_queries,
            'total_failures': len(failures),
            'failure_rate': len(failures) / total_queries if total_queries > 0 else 0,
            'patterns': patterns,
            'needs_review': needs_review,
            'needs_attention': needs_attention,
            'dominant_category': patterns[0].category if patterns else None,
        }
    
    def print_analysis_report(self, total_queries: int):
        """打印分析报告"""
        result = self.analyze_failure_patterns(total_queries)
        
        print("\n" + "="*70)
        print("Stage 12-A: 失效模式分析报告")
        print("="*70)
        
        if result['status'] == 'no_failures':
            print(f"\n  {result['message']}")
            print("="*70)
            return result
        
        print(f"\n  统计概览:")
        print(f"    总查询数: {result['total_queries']}")
        print(f"    失败案例: {result['total_failures']}")
        print(f"    失败率: {result['failure_rate']:.1%}")
        
        print(f"\n  失效模式分布:")
        for pattern in result['patterns']:
            risk_icon = "🔴" if pattern.risk_level == 'high' else "🟡" if pattern.risk_level == 'medium' else "🟢"
            print(f"\n    {risk_icon} {pattern.category.upper()}: {pattern.count}条 ({pattern.percentage:.1%})")
            print(f"       风险等级: {pattern.risk_level}")
            print(f"       示例:")
            for ex in pattern.examples:
                print(f"         - {ex}...")
        
        print(f"\n  预警状态:")
        if result['needs_review']:
            print(f"    🔴 高风险: 某类失败占比超过70%")
            print(f"    建议: 立即启动小型失效审查")
            print(f"    注意: 只做归因分类，不做在线训练！")
        elif result['needs_attention']:
            print(f"    🟡 中风险: 某类失败占比超过50%")
            print(f"    建议: 持续关注，下次分析时复查")
        else:
            print(f"    🟢 正常: 失败分布较为均衡")
        
        print("\n" + "="*70)
        
        return result
    
    def generate_mini_review_report(self) -> str:
        """生成小型失效审查报告"""
        failures = self.load_failure_cases()
        
        if not failures:
            return "暂无失败案例需要审查"
        
        # 按类别分组
        by_category = defaultdict(list)
        for case in failures:
            cat = case.get('category', 'unknown')
            by_category[cat].append(case)
        
        report = []
        report.append("="*70)
        report.append("Stage 12-A: 小型失效审查报告")
        report.append("="*70)
        report.append(f"\n审查时间: {datetime.now().isoformat()}")
        report.append(f"失败案例总数: {len(failures)}")
        report.append("\n审查原则:")
        report.append("  - 只做归因和分类整理")
        report.append("  - 不做在线增量训练")
        report.append("  - 为Stage 13积累知识")
        
        for cat, cases in sorted(by_category.items(), key=lambda x: -len(x[1])):
            report.append(f"\n{'='*70}")
            report.append(f"类别: {cat.upper()} ({len(cases)}条)")
            report.append("="*70)
            
            # 归因分析
            report.append("\n归因分析:")
            
            # 简单启发式归因
            if cat == 'h1':
                report.append("  - 可能原因: 伪科学/阴谋论/反科学表述识别不足")
                report.append("  - 典型特征: 事实错误但未触发H1拦截")
            elif cat == 'h2':
                report.append("  - 可能原因: 危险动作边界判断偏差")
                report.append("  - 典型特征: 恶意请求被误判为正常")
            elif cat == 'h4':
                report.append("  - 可能原因: 模糊表达泛化不足")
                report.append("  - 典型特征: 极度模糊查询未被拆分")
            elif cat == 'normal':
                report.append("  - 可能原因: 过度敏感/过度拦截")
                report.append("  - 典型特征: 正常查询被误判为危险")
            
            report.append("\n典型案例:")
            for i, case in enumerate(cases[:5], 1):  # 展示前5个
                report.append(f"\n  案例{i}:")
                report.append(f"    查询: {case.get('query', '')}")
                report.append(f"    期望: {case.get('expected_tsla', '')}")
                report.append(f"    实际: {case.get('predicted_tsla', '')}")
                report.append(f"    时间: {case.get('timestamp', '')}")
        
        report.append("\n" + "="*70)
        report.append("审查结论与建议")
        report.append("="*70)
        
        # 找出主导模式
        dominant = max(by_category.items(), key=lambda x: len(x[1]))
        dominant_cat, dominant_cases = dominant
        dominant_ratio = len(dominant_cases) / len(failures)
        
        report.append(f"\n主导失效模式: {dominant_cat.upper()} ({dominant_ratio:.1%})")
        
        if dominant_ratio >= 0.70:
            report.append("\n⚠️  警告: 单一失效模式占比过高！")
            report.append("建议:")
            report.append("  1. 继续收集更多案例，验证是否为系统性问题")
            report.append("  2. 在Stage 13中优先解决此类问题")
            report.append("  3. 考虑是否需要调整当前监控策略")
        elif dominant_ratio >= 0.50:
            report.append("\n🟡 关注: 单一失效模式占比较高")
            report.append("建议:")
            report.append("  1. 持续关注此类失败的发展")
            report.append("  2. 在Stage 13训练集中增加对应样本")
        else:
            report.append("\n🟢 正常: 失效模式分布较为均衡")
            report.append("建议:")
            report.append("  1. 继续按原计划收集案例")
            report.append("  2. 达到100条后启动Stage 13")
        
        report.append("\n" + "="*70)
        
        return "\n".join(report)


def run_failure_monitor_demo():
    """运行监控演示"""
    print("="*70)
    print("Stage 12-A: 失败案例监控与早期预警系统")
    print("="*70)
    
    monitor = FailureMonitor()
    
    # 模拟分析当前状态
    print("\n模拟场景: 已处理30条查询")
    result = monitor.print_analysis_report(total_queries=30)
    
    # 如果达到审查条件，生成审查报告
    if result.get('needs_review'):
        print("\n" + monitor.generate_mini_review_report())
    
    print("\n" + "="*70)
    print("监控规则总结")
    print("="*70)
    print("\n  自动触发条件:")
    print("    - 每30条查询: 自动失效模式分析")
    print("    - 某类失败>50%: 标记为'需关注'")
    print("    - 某类失败>70%: 触发小型失效审查")
    print("\n  审查原则:")
    print("    - 只做归因和分类整理")
    print("    - 不做在线增量训练")
    print("    - 为Stage 13积累知识")
    print("\n  当前状态:")
    print("    - 已收集3条失败案例")
    print("    - H1占主导(66.7%)")
    print("    - 建议: 继续监控，达到30条时复查")
    print("="*70)


if __name__ == "__main__":
    run_failure_monitor_demo()
