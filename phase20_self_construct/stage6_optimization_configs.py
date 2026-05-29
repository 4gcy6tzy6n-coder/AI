"""
Stage 6 Optimization Configurations

Phase 3 Task 4: 已知限制优化

当前限制:
- writeback 变化: 5.6-12% (目标 <5%)
- rollback 恢复率: 32.6% (目标 >90%)

优化方案:
1. 调整 KL 权重
2. 优化 replay 比例
3. 改进 rollback 快照策略
"""

from typing import Dict, List


# ==================== 官方基线 V1.0 ====================

OFFICIAL_BASELINE_V1 = {
    'version': '1.0',
    'name': 'Official Baseline',
    'base_kl_weights': {
        'gap': 0.30,
        'policy': 0.30,
        'governance': 0.30,
        'writeback': 0.42,
    },
    'learning_rate': 1.0e-5,
    'replay_ratio': 0.40,
    'step1_max_change': 0.003,
    'step2_max_change': 0.008,
    'old_ability_threshold': 0.12,
    'writeback_threshold': 0.05,
    'auto_rollback': True,
    'param_promotion_threshold': 0.80,
    'kb_promotion_threshold': 0.70,
}


# ==================== 优化方案 1: Writeback 保护优化 ====================

# 方案 1A: 增加 writeback KL 权重
WRITEBACK_CONFIG_A = {
    'version': '1.1a',
    'name': 'Writeback Protection A - Higher KL Weight',
    'description': '增加 writeback KL 权重到 0.50',
    'base_kl_weights': {
        'gap': 0.30,
        'policy': 0.30,
        'governance': 0.30,
        'writeback': 0.50,  # 增加
    },
    'learning_rate': 1.0e-5,
    'replay_ratio': 0.40,
    'step1_max_change': 0.003,
    'step2_max_change': 0.008,
    'old_ability_threshold': 0.12,
    'writeback_threshold': 0.05,
    'auto_rollback': True,
    'param_promotion_threshold': 0.80,
    'kb_promotion_threshold': 0.70,
}

# 方案 1B: 增加 replay 比例
WRITEBACK_CONFIG_B = {
    'version': '1.1b',
    'name': 'Writeback Protection B - Higher Replay Ratio',
    'description': '增加 replay 比例到 0.50',
    'base_kl_weights': {
        'gap': 0.30,
        'policy': 0.30,
        'governance': 0.30,
        'writeback': 0.42,
    },
    'learning_rate': 1.0e-5,
    'replay_ratio': 0.50,  # 增加
    'step1_max_change': 0.003,
    'step2_max_change': 0.008,
    'old_ability_threshold': 0.12,
    'writeback_threshold': 0.05,
    'auto_rollback': True,
    'param_promotion_threshold': 0.80,
    'kb_promotion_threshold': 0.70,
}

# 方案 1C: 组合优化
WRITEBACK_CONFIG_C = {
    'version': '1.1c',
    'name': 'Writeback Protection C - Combined',
    'description': 'KL 权重 0.50 + replay 比例 0.50',
    'base_kl_weights': {
        'gap': 0.30,
        'policy': 0.30,
        'governance': 0.30,
        'writeback': 0.50,
    },
    'learning_rate': 1.0e-5,
    'replay_ratio': 0.50,
    'step1_max_change': 0.003,
    'step2_max_change': 0.008,
    'old_ability_threshold': 0.12,
    'writeback_threshold': 0.05,
    'auto_rollback': True,
    'param_promotion_threshold': 0.80,
    'kb_promotion_threshold': 0.70,
}


# ==================== 优化方案 2: Rollback 恢复优化 ====================

# 方案 2A: 降低回滚阈值
ROLLBACK_CONFIG_A = {
    'version': '1.2a',
    'name': 'Rollback Optimization A - Lower Thresholds',
    'description': '降低晋升阈值，减少激进晋升',
    'base_kl_weights': {
        'gap': 0.30,
        'policy': 0.30,
        'governance': 0.30,
        'writeback': 0.42,
    },
    'learning_rate': 1.0e-5,
    'replay_ratio': 0.40,
    'step1_max_change': 0.003,
    'step2_max_change': 0.008,
    'old_ability_threshold': 0.10,  # 降低
    'writeback_threshold': 0.04,    # 降低
    'auto_rollback': True,
    'param_promotion_threshold': 0.85,  # 提高
    'kb_promotion_threshold': 0.75,     # 提高
}

# 方案 2B: 激进回滚策略
ROLLBACK_CONFIG_B = {
    'version': '1.2b',
    'name': 'Rollback Optimization B - Aggressive Rollback',
    'description': '更激进的回滚策略',
    'base_kl_weights': {
        'gap': 0.30,
        'policy': 0.30,
        'governance': 0.30,
        'writeback': 0.42,
    },
    'learning_rate': 1.0e-5,
    'replay_ratio': 0.40,
    'step1_max_change': 0.002,  # 更保守
    'step2_max_change': 0.005,  # 更保守
    'old_ability_threshold': 0.08,   # 更严格
    'writeback_threshold': 0.03,     # 更严格
    'auto_rollback': True,
    'param_promotion_threshold': 0.75,
    'kb_promotion_threshold': 0.65,
}


# ==================== 优化方案 3: 综合优化 ====================

OPTIMIZED_CONFIG_V2 = {
    'version': '2.0',
    'name': 'Optimized V2.0',
    'description': '综合优化 writeback 和 rollback',
    'base_kl_weights': {
        'gap': 0.30,
        'policy': 0.30,
        'governance': 0.30,
        'writeback': 0.50,  # 增加保护
    },
    'learning_rate': 8.0e-6,  # 降低学习率
    'replay_ratio': 0.50,     # 增加 replay
    'step1_max_change': 0.002,  # 更保守
    'step2_max_change': 0.005,
    'old_ability_threshold': 0.10,
    'writeback_threshold': 0.04,
    'auto_rollback': True,
    'param_promotion_threshold': 0.85,
    'kb_promotion_threshold': 0.75,
}


# ==================== 配置集合 ====================

ALL_CONFIGS = {
    'baseline_v1': OFFICIAL_BASELINE_V1,
    'writeback_a': WRITEBACK_CONFIG_A,
    'writeback_b': WRITEBACK_CONFIG_B,
    'writeback_c': WRITEBACK_CONFIG_C,
    'rollback_a': ROLLBACK_CONFIG_A,
    'rollback_b': ROLLBACK_CONFIG_B,
    'optimized_v2': OPTIMIZED_CONFIG_V2,
}


# ==================== 配置评估目标 ====================

OPTIMIZATION_TARGETS = {
    'writeback_change': {
        'current': 0.10,  # 10%
        'target': 0.05,   # 5%
        'priority': 'high',
    },
    'rollback_recovery': {
        'current': 0.326,  # 32.6%
        'target': 0.90,    # 90%
        'priority': 'high',
    },
    'target_gain': {
        'current': 0.12,  # 12%
        'target': 0.10,   # 10%
        'priority': 'must',
    },
    'old_ability_drop': {
        'current': 0.03,  # 3%
        'target': 0.15,   # 15%
        'priority': 'must',
    },
}


# ==================== 配置选择函数 ====================

def get_config(config_name: str) -> Dict:
    """获取指定配置"""
    return ALL_CONFIGS.get(config_name, OFFICIAL_BASELINE_V1)


def get_all_configs() -> Dict[str, Dict]:
    """获取所有配置"""
    return ALL_CONFIGS


def compare_configs(config1_name: str, config2_name: str) -> Dict:
    """比较两个配置"""
    config1 = get_config(config1_name)
    config2 = get_config(config2_name)
    
    differences = {}
    
    for key in config1:
        if key in config2 and config1[key] != config2[key]:
            differences[key] = {
                'config1': config1[key],
                'config2': config2[key],
            }
    
    return differences


def print_config_summary(config_name: str = 'baseline_v1'):
    """打印配置摘要"""
    config = get_config(config_name)
    
    print(f"\n{'='*70}")
    print(f"配置: {config.get('name', config_name)} (v{config.get('version', 'N/A')})")
    print(f"{'='*70}")
    
    print(f"\nKL Weights:")
    for key, value in config.get('base_kl_weights', {}).items():
        print(f"  {key}: {value}")
    
    print(f"\n其他参数:")
    print(f"  learning_rate: {config.get('learning_rate')}")
    print(f"  replay_ratio: {config.get('replay_ratio')}")
    print(f"  step1_max_change: {config.get('step1_max_change')}")
    print(f"  step2_max_change: {config.get('step2_max_change')}")
    print(f"  old_ability_threshold: {config.get('old_ability_threshold')}")
    print(f"  writeback_threshold: {config.get('writeback_threshold')}")
    print(f"  param_promotion_threshold: {config.get('param_promotion_threshold')}")
    print(f"  kb_promotion_threshold: {config.get('kb_promotion_threshold')}")
    
    print(f"\n{'='*70}")


def print_all_configs_summary():
    """打印所有配置摘要"""
    print(f"\n{'='*70}")
    print("所有优化配置摘要")
    print(f"{'='*70}")
    
    for name, config in ALL_CONFIGS.items():
        print(f"\n{name}:")
        print(f"  版本: {config.get('version')}")
        print(f"  名称: {config.get('name')}")
        print(f"  描述: {config.get('description', 'N/A')}")
        print(f"  writeback KL: {config['base_kl_weights']['writeback']}")
        print(f"  replay_ratio: {config['replay_ratio']}")


# ==================== 测试 ====================

if __name__ == "__main__":
    print_all_configs_summary()
    
    print("\n\n配置比较 (baseline vs optimized_v2):")
    diff = compare_configs('baseline_v1', 'optimized_v2')
    for key, values in diff.items():
        print(f"  {key}: {values['config1']} -> {values['config2']}")
