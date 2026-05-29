import random
from typing import Any


class NoiseConfig:
    """噪声配置"""

    def __init__(
        self,
        input_noise: dict[str, float] | None = None,
        retrieval_noise: dict[str, float] | None = None,
        reasoning_noise: dict[str, float] | None = None,
        intensity: float = 0.3
    ):
        self.input_noise = input_noise or {
            "typo_rate": 0.1,
            "word_dropout_rate": 0.05,
            "shuffle_rate": 0.02
        }
        self.retrieval_noise = retrieval_noise or {
            "irrelevant_result_rate": 0.2,
            "missing_result_rate": 0.1,
            "ranking_error_rate": 0.15
        }
        self.reasoning_noise = reasoning_noise or {
            "step_dropout_rate": 0.1,
            "wrong_step_rate": 0.05,
            "circular_reasoning_rate": 0.02
        }
        self.intensity = intensity


class NoiseInjector:
    """噪声注入器"""

    def __init__(self, config: NoiseConfig | None = None):
        self.config = config or NoiseConfig()

    def inject_input_noise(self, content: str) -> str:
        """注入输入噪声"""
        words = content.split()
        noisy_words = []

        for word in words:
            # 词语丢弃
            if random.random() < self.config.input_noise.get("word_dropout_rate", 0.05):
                continue

            # 错别字注入
            if random.random() < self.config.input_noise.get("typo_rate", 0.1):
                word = self._inject_typo(word)

            noisy_words.append(word)

        # 顺序打乱
        if random.random() < self.config.input_noise.get("shuffle_rate", 0.02):
            noisy_words = self._shuffle_words(noisy_words)

        return " ".join(noisy_words)

    def inject_retrieval_noise(
        self,
        results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """注入检索噪声"""
        noisy_results = results.copy()

        # 添加无关结果
        if random.random() < self.config.retrieval_noise.get("irrelevant_result_rate", 0.2):
            noisy_results.append({
                "result_id": "noise_1",
                "content": "Irrelevant noise content",
                "relevance_score": 0.1
            })

        # 移除结果
        if random.random() < self.config.retrieval_noise.get("missing_result_rate", 0.1):
            if noisy_results:
                noisy_results.pop(random.randint(0, len(noisy_results) - 1))

        # 排序错误
        if random.random() < self.config.retrieval_noise.get("ranking_error_rate", 0.15):
            random.shuffle(noisy_results)

        return noisy_results

    def inject_reasoning_noise(
        self,
        steps: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """注入推理噪声"""
        noisy_steps = steps.copy()

        # 步骤丢弃
        if random.random() < self.config.reasoning_noise.get("step_dropout_rate", 0.1):
            if len(noisy_steps) > 1:
                noisy_steps.pop(random.randint(0, len(noisy_steps) - 1))

        # 错误步骤
        if random.random() < self.config.reasoning_noise.get("wrong_step_rate", 0.05):
            if noisy_steps:
                idx = random.randint(0, len(noisy_steps) - 1)
                noisy_steps[idx]["description"] = "Wrong step: " + noisy_steps[idx].get("description", "")

        return noisy_steps

    def _inject_typo(self, word: str) -> str:
        """注入错别字"""
        if len(word) < 2:
            return word

        # 简单的错别字：交换相邻字符
        chars = list(word)
        idx = random.randint(0, len(chars) - 2)
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        return "".join(chars)

    def _shuffle_words(self, words: list[str]) -> list[str]:
        """打乱词语顺序"""
        words = words.copy()
        # 只打乱一小部分
        if len(words) >= 2:
            idx = random.randint(0, len(words) - 2)
            words[idx], words[idx + 1] = words[idx + 1], words[idx]
        return words
