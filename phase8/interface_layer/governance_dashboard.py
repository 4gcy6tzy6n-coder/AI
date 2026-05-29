"""
Governance Dashboard - 治理系统可视化仪表板

WP4 核心组件：
提供治理系统的可视化监控界面

功能：
1. 实时治理指标展示
2. Unit 状态可视化
3. 系统健康度监控
4. 历史趋势分析
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class DashboardMetrics:
    """仪表板指标"""
    qt_score: float
    sl_score: float
    t_score: float
    c_score: float
    l_score: float
    overall_health: float
    active_units: int
    pending_governance: int
    error_rate: float
    avg_latency_ms: float


@dataclass
class UnitStatus:
    """Unit 状态"""
    unit_id: str
    unit_type: str
    quality_tier: str  # high/medium/low
    stability_status: str  # stable/unstable/pending
    last_updated: str
    governance_score: float


class GovernanceDashboard:
    """
    治理系统仪表板
    
    功能：
    1. 生成可视化报告
    2. 展示系统健康度
    3. 显示 Unit 分布
    4. 提供趋势分析
    """
    
    def __init__(self):
        self.metrics_history = []
        self._generate_sample_data()
    
    def _generate_sample_data(self):
        """生成示例数据"""
        # 生成历史指标数据
        base_time = datetime.now() - timedelta(hours=24)
        for i in range(24):
            timestamp = base_time + timedelta(hours=i)
            self.metrics_history.append({
                "timestamp": timestamp.isoformat(),
                "qt_score": random.uniform(0.75, 0.92),
                "sl_score": random.uniform(0.80, 0.95),
                "t_score": random.uniform(0.70, 0.88),
                "c_score": random.uniform(0.78, 0.93),
                "l_score": random.uniform(0.82, 0.94),
                "unit_count": random.randint(8000, 12000),
                "error_rate": random.uniform(0.01, 0.08)
            })
    
    def get_current_metrics(self) -> DashboardMetrics:
        """获取当前指标"""
        return DashboardMetrics(
            qt_score=random.uniform(0.80, 0.92),
            sl_score=random.uniform(0.85, 0.95),
            t_score=random.uniform(0.75, 0.88),
            c_score=random.uniform(0.80, 0.92),
            l_score=random.uniform(0.85, 0.94),
            overall_health=random.uniform(0.82, 0.93),
            active_units=random.randint(10000, 15000),
            pending_governance=random.randint(50, 200),
            error_rate=random.uniform(0.02, 0.06),
            avg_latency_ms=random.uniform(40, 60)
        )
    
    def get_unit_distribution(self) -> Dict[str, Any]:
        """获取 Unit 分布"""
        return {
            "by_type": {
                "concept": random.randint(5000, 7000),
                "relation": random.randint(2000, 3500),
                "rule": random.randint(500, 1000),
                "task_pattern": random.randint(800, 1500)
            },
            "by_quality": {
                "high": random.randint(6000, 8000),
                "medium": random.randint(3000, 5000),
                "low": random.randint(500, 1500)
            },
            "by_stability": {
                "stable": random.randint(8000, 10000),
                "unstable": random.randint(500, 1500),
                "pending": random.randint(1000, 2500)
            }
        }
    
    def get_recent_units(self, limit: int = 10) -> List[UnitStatus]:
        """获取最近的 Units"""
        units = []
        unit_types = ["concept", "relation", "rule", "task_pattern"]
        quality_tiers = ["high", "medium", "low"]
        stability_statuses = ["stable", "unstable", "pending"]
        
        for i in range(limit):
            units.append(UnitStatus(
                unit_id=f"unit_{random.randint(10000, 99999)}",
                unit_type=random.choice(unit_types),
                quality_tier=random.choice(quality_tiers),
                stability_status=random.choice(stability_statuses),
                last_updated=(datetime.now() - timedelta(minutes=random.randint(1, 60))).isoformat(),
                governance_score=random.uniform(0.6, 0.95)
            ))
        
        return units
    
    def get_system_health(self) -> Dict[str, Any]:
        """获取系统健康度"""
        metrics = self.get_current_metrics()
        
        # 计算各维度健康度
        health_breakdown = {
            "quality": {
                "score": metrics.qt_score,
                "status": "healthy" if metrics.qt_score >= 0.8 else "warning" if metrics.qt_score >= 0.6 else "critical",
                "icon": "🟢" if metrics.qt_score >= 0.8 else "🟡" if metrics.qt_score >= 0.6 else "🔴"
            },
            "stability": {
                "score": metrics.sl_score,
                "status": "healthy" if metrics.sl_score >= 0.8 else "warning" if metrics.sl_score >= 0.6 else "critical",
                "icon": "🟢" if metrics.sl_score >= 0.8 else "🟡" if metrics.sl_score >= 0.6 else "🔴"
            },
            "timeliness": {
                "score": metrics.t_score,
                "status": "healthy" if metrics.t_score >= 0.7 else "warning" if metrics.t_score >= 0.5 else "critical",
                "icon": "🟢" if metrics.t_score >= 0.7 else "🟡" if metrics.t_score >= 0.5 else "🔴"
            },
            "consistency": {
                "score": metrics.c_score,
                "status": "healthy" if metrics.c_score >= 0.8 else "warning" if metrics.c_score >= 0.6 else "critical",
                "icon": "🟢" if metrics.c_score >= 0.8 else "🟡" if metrics.c_score >= 0.6 else "🔴"
            },
            "legitimacy": {
                "score": metrics.l_score,
                "status": "healthy" if metrics.l_score >= 0.8 else "warning" if metrics.l_score >= 0.6 else "critical",
                "icon": "🟢" if metrics.l_score >= 0.8 else "🟡" if metrics.l_score >= 0.6 else "🔴"
            }
        }
        
        # 整体健康度
        overall_status = "healthy"
        if any(h["status"] == "critical" for h in health_breakdown.values()):
            overall_status = "critical"
        elif any(h["status"] == "warning" for h in health_breakdown.values()):
            overall_status = "warning"
        
        return {
            "overall": {
                "score": metrics.overall_health,
                "status": overall_status,
                "icon": "🟢" if overall_status == "healthy" else "🟡" if overall_status == "warning" else "🔴"
            },
            "breakdown": health_breakdown,
            "last_updated": datetime.now().isoformat()
        }
    
    def get_trend_analysis(self, hours: int = 24) -> Dict[str, Any]:
        """获取趋势分析"""
        # 使用历史数据计算趋势
        recent_data = self.metrics_history[-hours:] if len(self.metrics_history) >= hours else self.metrics_history
        
        if len(recent_data) < 2:
            return {"error": "Insufficient data for trend analysis"}
        
        # 计算平均值
        avg_qt = sum(d["qt_score"] for d in recent_data) / len(recent_data)
        avg_sl = sum(d["sl_score"] for d in recent_data) / len(recent_data)
        avg_error = sum(d["error_rate"] for d in recent_data) / len(recent_data)
        
        # 计算趋势（简单线性）
        first_half = recent_data[:len(recent_data)//2]
        second_half = recent_data[len(recent_data)//2:]
        
        first_qt = sum(d["qt_score"] for d in first_half) / len(first_half)
        second_qt = sum(d["qt_score"] for d in second_half) / len(second_half)
        
        qt_trend = "improving" if second_qt > first_qt else "declining" if second_qt < first_qt else "stable"
        
        return {
            "period_hours": hours,
            "averages": {
                "qt_score": avg_qt,
                "sl_score": avg_sl,
                "error_rate": avg_error
            },
            "trends": {
                "qt_trend": qt_trend,
                "qt_change": second_qt - first_qt
            },
            "data_points": len(recent_data)
        }
    
    def render_dashboard(self):
        """渲染仪表板"""
        print("\n" + "📊 " * 35)
        print("Governance Dashboard - 治理系统仪表板")
        print("📊 " * 35)
        
        # 1. 系统健康度
        print("\n" + "="*70)
        print("1. 系统健康度")
        print("="*70)
        
        health = self.get_system_health()
        overall = health["overall"]
        
        print(f"\n  整体健康度: {overall['icon']} {overall['score']:.2f} ({overall['status'].upper()})")
        print(f"\n  各维度健康度:")
        for dimension, data in health["breakdown"].items():
            bar = "█" * int(data["score"] * 20)
            print(f"    {data['icon']} {dimension:12} {data['score']:.2f} {bar}")
        
        # 2. 当前指标
        print("\n" + "="*70)
        print("2. 当前治理指标")
        print("="*70)
        
        metrics = self.get_current_metrics()
        print(f"\n  QT (质量):     {metrics.qt_score:.2f} █{'█' * int(metrics.qt_score * 10)}")
        print(f"  SL (稳定性):   {metrics.sl_score:.2f} █{'█' * int(metrics.sl_score * 10)}")
        print(f"  T (时效性):    {metrics.t_score:.2f} █{'█' * int(metrics.t_score * 10)}")
        print(f"  C (一致性):    {metrics.c_score:.2f} █{'█' * int(metrics.c_score * 10)}")
        print(f"  L (合法性):    {metrics.l_score:.2f} █{'█' * int(metrics.l_score * 10)}")
        
        print(f"\n  活跃 Units:    {metrics.active_units:,}")
        print(f"  待治理:        {metrics.pending_governance}")
        print(f"  错误率:        {metrics.error_rate:.1%}")
        print(f"  平均延迟:      {metrics.avg_latency_ms:.1f} ms")
        
        # 3. Unit 分布
        print("\n" + "="*70)
        print("3. Unit 分布")
        print("="*70)
        
        distribution = self.get_unit_distribution()
        
        print(f"\n  按类型分布:")
        total = sum(distribution["by_type"].values())
        for unit_type, count in distribution["by_type"].items():
            percentage = count / total * 100
            bar = "█" * int(percentage / 5)
            print(f"    {unit_type:15} {count:6,} ({percentage:5.1f}%) {bar}")
        
        print(f"\n  按质量分布:")
        for quality, count in distribution["by_quality"].items():
            percentage = count / total * 100
            icon = "🟢" if quality == "high" else "🟡" if quality == "medium" else "🔴"
            print(f"    {icon} {quality:10} {count:6,} ({percentage:5.1f}%)")
        
        print(f"\n  按稳定性分布:")
        for stability, count in distribution["by_stability"].items():
            percentage = count / total * 100
            icon = "🟢" if stability == "stable" else "🟡" if stability == "pending" else "🔴"
            print(f"    {icon} {stability:10} {count:6,} ({percentage:5.1f}%)")
        
        # 4. 最近 Units
        print("\n" + "="*70)
        print("4. 最近更新的 Units")
        print("="*70)
        
        recent_units = self.get_recent_units(5)
        print(f"\n  {'Unit ID':<15} {'类型':<12} {'质量':<8} {'稳定性':<10} {'分数':<6}")
        print(f"  {'-'*60}")
        for unit in recent_units:
            quality_icon = "🟢" if unit.quality_tier == "high" else "🟡" if unit.quality_tier == "medium" else "🔴"
            stability_icon = "🟢" if unit.stability_status == "stable" else "🟡" if unit.stability_status == "pending" else "🔴"
            print(f"  {unit.unit_id:<15} {unit.unit_type:<12} {quality_icon} {unit.quality_tier:<6} {stability_icon} {unit.stability_status:<8} {unit.governance_score:.2f}")
        
        # 5. 趋势分析
        print("\n" + "="*70)
        print("5. 趋势分析 (最近24小时)")
        print("="*70)
        
        trend = self.get_trend_analysis(24)
        print(f"\n  平均值:")
        print(f"    QT分数: {trend['averages']['qt_score']:.3f}")
        print(f"    SL分数: {trend['averages']['sl_score']:.3f}")
        print(f"    错误率: {trend['averages']['error_rate']:.3f}")
        
        print(f"\n  趋势:")
        trend_icon = "📈" if trend['trends']['qt_trend'] == "improving" else "📉" if trend['trends']['qt_trend'] == "declining" else "➡️"
        print(f"    {trend_icon} QT趋势: {trend['trends']['qt_trend']} ({trend['trends']['qt_change']:+.4f})")
        
        print("\n" + "="*70)
        print("仪表板渲染完成")
        print("="*70)


def demo_governance_dashboard():
    """演示治理仪表板"""
    dashboard = GovernanceDashboard()
    dashboard.render_dashboard()


if __name__ == "__main__":
    demo_governance_dashboard()
