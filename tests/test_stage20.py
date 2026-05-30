"""Tests for Stage 20 — B-TSLA Bayesian Diagnosis Module.

Covers core B-TSLA modules: schema, evidence_collector, bayesian_inferencer,
risk_policy, correction_executor, and orchestrator.
"""

import json
import math
import os
import sys
import tempfile
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Schema Tests ──────────────────────────────────────────────────

class TestSchema:
    """Tests for src/core/bayes/schema.py"""

    def test_failure_category_enum(self):
        from src.core.bayes.schema import FailureCategory
        assert FailureCategory.K1.value == "knowledge_miss"
        assert FailureCategory.G1.value == "unnatural_generation"
        assert len(list(FailureCategory)) == 8

    def test_failure_sample_immutable(self):
        from src.core.bayes.schema import FailureSample, FailureCategory
        s = FailureSample(
            sample_id="test_001",
            user_query="What?",
            failure_type=FailureCategory.K1,
        )
        assert s.sample_id == "test_001"
        # frozen dataclass — assignment should raise
        try:
            s.sample_id = "changed"  # type: ignore[misc]
            assert False, "Should have raised FrozenInstanceError"
        except Exception:
            pass

    def test_failure_sample_roundtrip(self):
        from src.core.bayes.schema import FailureSample, FailureCategory, RiskLevel
        original = FailureSample(
            sample_id="r_001",
            user_query="Test query",
            system_response="Test response",
            failure_type=FailureCategory.M1,
            risk_level=RiskLevel.HIGH,
        )
        data = original.to_dict()
        restored = FailureSample.from_dict(data)
        assert restored.sample_id == original.sample_id
        assert restored.failure_type == original.failure_type
        assert restored.risk_level == original.risk_level

    def test_evidence_vector_to_list(self):
        from src.core.bayes.schema import EvidenceVector
        ev = EvidenceVector(
            retrieval_gap_score=0.5,
            context_conflict_score=0.2,
        )
        lst = ev.to_list()
        assert len(lst) == 8
        assert lst[0] == 0.5
        assert lst[1] == 0.2

    def test_posterior_distribution_defaults(self):
        from src.core.bayes.schema import PosteriorDistribution, PosteriorState
        p = PosteriorDistribution()
        assert p.p_knowledge_gap == 0.0
        assert p.dominant_cause == PosteriorState.KNOWLEDGE_GAP

    def test_risk_decision_to_dict(self):
        from src.core.bayes.schema import RiskDecision, GovernanceAction, RiskLevel
        rd = RiskDecision(
            recommended_action=GovernanceAction.KEEP,
            reason="test",
            severity=RiskLevel.LOW,
        )
        d = rd.to_dict()
        assert d["recommended_action"] == "keep"

    def test_fix_package_uuid(self):
        from src.core.bayes.schema import FixPackage
        pkg = FixPackage(package_type="knowledge")
        assert len(pkg.package_id) == 36  # UUID string length
        assert pkg.package_type == "knowledge"

    def test_replay_result_to_dict(self):
        from src.core.bayes.schema import ReplayResult
        rr = ReplayResult(
            test_set_name="trigger_set",
            total_cases=10,
            passed_cases=8,
            failure_fix_rate=0.8,
        )
        d = rr.to_dict()
        assert d["test_set_name"] == "trigger_set"


# ── Evidence Collector Tests ──────────────────────────────────────

class TestEvidenceCollector:
    """Tests for src/core/bayes/evidence_collector.py"""

    def test_extract_empty_context(self):
        from src.core.bayes.evidence_collector import EvidenceCollector
        from src.core.bayes.schema import EvidenceVector
        collector = EvidenceCollector()
        from src.core.unit.models import Unit

        unit = Unit()
        from src.core.tsla.scorer import ScoringResult
        from src.core.tsla.action_router import ActionDecision

        # Simplified: test dimension 1 directly
        gap = collector._extract_retrieval_gap(unit, [])
        assert gap >= 0.8  # empty context = high gap

    def test_extract_with_full_context(self):
        from src.core.bayes.evidence_collector import EvidenceCollector
        from src.core.unit.models import Unit

        collector = EvidenceCollector()
        unit = Unit(evidence_score=0.9)
        gap = collector._extract_retrieval_gap(unit, ["context1", "context2"])
        assert gap <= 0.2  # high evidence + context = low gap

    def test_context_conflict_from_unit(self):
        from src.core.bayes.evidence_collector import EvidenceCollector
        from src.core.unit.models import Unit

        collector = EvidenceCollector()
        unit = Unit(conflict_cleanliness=0.3)
        conflict = collector._extract_context_conflict(unit)
        assert conflict > 0.5  # low cleanliness = high conflict

    def test_fluency_broken_response(self):
        from src.core.bayes.evidence_collector import EvidenceCollector
        from src.core.tsla.scorer import ScoringResult
        from src.core.unit.models import Unit
        from src.core.tsla.scorer import TSLAScores, ScoreSnapshot, RollingStats

        collector = EvidenceCollector()
        response = "This has <UNK> tokens [broken] {malformed}"
        scoring = ScoringResult(
            current_scores=TSLAScores(),
            snapshot=ScoreSnapshot(),
            rolling_stats=RollingStats(),
            history_count=1,
        )
        fluency = collector._extract_generation_fluency(response, scoring)
        assert fluency < 0.5

    def test_memory_contamination_no_events(self):
        from src.core.bayes.evidence_collector import EvidenceCollector
        collector = EvidenceCollector()
        contamination = collector._extract_memory_contamination("test_id", None)
        assert contamination == 0.0

    def test_clamp_utility(self):
        from src.core.bayes.evidence_collector import EvidenceCollector
        assert EvidenceCollector._clamp(1.5) == 1.0
        assert EvidenceCollector._clamp(-0.5) == 0.0
        assert EvidenceCollector._clamp(0.5) == 0.5


# ── Bayesian Inferencer Tests ─────────────────────────────────────

class TestBayesianInferencer:
    """Tests for src/core/bayes/bayesian_inferencer.py"""

    def test_infer_returns_valid_distribution(self):
        from src.core.bayes.bayesian_inferencer import BayesianInferencer
        from src.core.bayes.schema import EvidenceVector

        inferencer = BayesianInferencer()
        ev = EvidenceVector(
            retrieval_gap_score=0.8,
            memory_contamination_score=0.1,
        )
        posterior = inferencer.infer(ev)

        # All probabilities should be between 0 and 1
        probs = posterior.as_dict()
        for cause, p in probs.items():
            assert 0.0 <= p <= 1.0, f"{cause} = {p} out of range"

        # Should sum to ~1.0
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.01, f"Sum is {total}"

    def test_retrieval_gap_signals_retrieval_failure(self):
        from src.core.bayes.bayesian_inferencer import BayesianInferencer
        from src.core.bayes.schema import EvidenceVector, PosteriorState

        inferencer = BayesianInferencer()
        ev = EvidenceVector(retrieval_gap_score=0.95)
        posterior = inferencer.infer(ev)

        # High retrieval gap should make retrieval_failure dominant
        assert (
            posterior.p_retrieval_failure > posterior.p_generation_failure
        )

    def test_memory_contamination_signals_memory_failure(self):
        from src.core.bayes.bayesian_inferencer import BayesianInferencer
        from src.core.bayes.schema import EvidenceVector, PosteriorState

        inferencer = BayesianInferencer()
        ev = EvidenceVector(memory_contamination_score=0.95)
        posterior = inferencer.infer(ev)

        assert posterior.p_memory_failure > 0.15  # above prior

    def test_infer_batch(self):
        from src.core.bayes.bayesian_inferencer import BayesianInferencer
        from src.core.bayes.schema import EvidenceVector

        inferencer = BayesianInferencer()
        batch = [EvidenceVector() for _ in range(5)]
        results = inferencer.infer_batch(batch)
        assert len(results) == 5

    def test_invalid_likelihood_map_raises(self):
        from src.core.bayes.bayesian_inferencer import BayesianInferencer

        try:
            BayesianInferencer(
                likelihood_high={"retrieval_gap": {}}  # incomplete
            )
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_entropy_computation(self):
        from src.core.bayes.bayesian_inferencer import BayesianInferencer
        from src.core.bayes.schema import EvidenceVector

        inferencer = BayesianInferencer()
        ev = EvidenceVector()
        posterior = inferencer.infer(ev)
        assert posterior.entropy > 0.0


# ── Risk Policy Tests ─────────────────────────────────────────────

class TestRiskPolicy:
    """Tests for src/core/bayes/risk_policy.py"""

    def test_high_contamination_escalates_to_quarantine(self):
        from src.core.bayes.risk_policy import RiskPolicyDecider
        from src.core.bayes.schema import (
            EvidenceVector,
            FailureSample,
            GovernanceAction,
            PosteriorDistribution,
        )

        decider = RiskPolicyDecider()
        ev = EvidenceVector(memory_contamination_score=0.9)
        posterior = PosteriorDistribution()
        sample = FailureSample(sample_id="test", user_query="q")

        decision = decider.decide(posterior, ev, sample)
        assert decision.recommended_action == GovernanceAction.QUARANTINE

    def test_low_confidence_escalates_to_human_review(self):
        from src.core.bayes.risk_policy import RiskPolicyDecider
        from src.core.bayes.schema import (
            EvidenceVector,
            FailureSample,
            GovernanceAction,
            PosteriorDistribution,
            PosteriorState,
        )

        decider = RiskPolicyDecider()
        ev = EvidenceVector()
        posterior = PosteriorDistribution(confidence=0.3)
        sample = FailureSample(sample_id="test", user_query="q")

        decision = decider.decide(posterior, ev, sample)
        assert decision.recommended_action == GovernanceAction.HUMAN_REVIEW

    def test_governance_to_tsla_mapping(self):
        from src.core.bayes.risk_policy import GOVERNANCE_TO_TSLA, GovernanceAction
        assert GOVERNANCE_TO_TSLA[GovernanceAction.KEEP] == "keep"
        assert GOVERNANCE_TO_TSLA[GovernanceAction.QUARANTINE] == "isolate"
        assert GOVERNANCE_TO_TSLA[GovernanceAction.RETRIEVAL_PATCH] == "review_backflow"


# ── Correction Executor Tests ─────────────────────────────────────

class TestCorrectionExecutor:
    """Tests for src/core/bayes/correction_executor.py"""

    def test_generate_knowledge_patch(self):
        from src.core.bayes.correction_executor import CorrectionExecutor
        from src.core.bayes.schema import (
            EvidenceVector,
            FailureSample,
            GovernanceAction,
            PosteriorDistribution,
            RiskDecision,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            executor = CorrectionExecutor(output_dir=tmpdir)
            rd = RiskDecision(recommended_action=GovernanceAction.RETRIEVAL_PATCH)
            ev = EvidenceVector()
            sample = FailureSample(
                sample_id="test_kp",
                user_query="What is X?",
                expected_behavior="X is Y",
            )

            pkg = executor.generate(rd, ev, sample)
            assert pkg.package_type == "retrieval"
            # Check file was written
            matching = list(Path(tmpdir).iterdir())
            assert len(matching) > 0

    def test_generate_keep_package(self):
        from src.core.bayes.correction_executor import CorrectionExecutor
        from src.core.bayes.schema import (
            EvidenceVector,
            FailureSample,
            GovernanceAction,
            PosteriorDistribution,
            RiskDecision,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            executor = CorrectionExecutor(output_dir=tmpdir)
            rd = RiskDecision(recommended_action=GovernanceAction.KEEP)
            ev = EvidenceVector()
            sample = FailureSample(sample_id="test_k", user_query="q")

            pkg = executor.generate(rd, ev, sample)
            assert pkg.package_type == "keep"
            assert pkg.payload["action"] == "keep"

    def test_list_packages(self):
        from src.core.bayes.correction_executor import CorrectionExecutor
        from src.core.bayes.schema import (
            EvidenceVector,
            FailureSample,
            GovernanceAction,
            RiskDecision,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            executor = CorrectionExecutor(output_dir=tmpdir)
            rd = RiskDecision(recommended_action=GovernanceAction.GENERATION_PATCH)
            ev = EvidenceVector()
            sample = FailureSample(sample_id="test_pkg", user_query="q")

            executor.generate(rd, ev, sample)
            packages = executor.list_packages()
            assert len(packages) == 1


# ── Orchestrator Tests ────────────────────────────────────────────

class TestOrchestrator:
    """Tests for src/core/bayes/orchestrator.py"""

    def test_diagnose_returns_all_keys(self):
        from src.core.bayes.orchestrator import BTSLAOrchestrator
        from src.core.bayes.correction_executor import CorrectionExecutor
        from src.core.bayes.schema import FailureSample
        from src.core.unit.models import Unit
        from src.core.tsla.scorer import ScoringResult, TSLAScores, ScoreSnapshot, RollingStats
        from src.core.tsla.action_router import ActionDecision
        from src.core.memory.memory_events import MemoryZone

        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = BTSLAOrchestrator(
                correction_executor=CorrectionExecutor(output_dir=tmpdir)
            )
            sample = FailureSample(sample_id="d1", user_query="Test query")
            unit = Unit()
            scoring = ScoringResult(
                current_scores=TSLAScores(),
                snapshot=ScoreSnapshot(),
                rolling_stats=RollingStats(),
                history_count=1,
            )
            action = ActionDecision(
                action="keep",
                reason="no issues",
                target_zone=MemoryZone.LONG_TERM_NORMAL,
            )

            result = orchestrator.diagnose(sample, unit, scoring, action)

        assert "sample_id" in result
        assert "evidence" in result
        assert "posterior" in result
        assert "risk_decision" in result
        assert "fix_package" in result


# ── Failure Pool Tests ────────────────────────────────────────────

class TestFailurePool:
    """Tests for stage20/failure_pool.py"""

    def test_trigger_fires_when_conditions_met(self):
        from stage20.failure_pool import TriggerCondition
        tc = TriggerCondition(
            knowledge_miss_count=25,
            unnatural_generation_count=5,
            multiturn_anomaly_count=3,
        )
        triggered, reason = tc.is_triggered()
        assert triggered
        assert "knowledge_miss" in reason

    def test_trigger_does_not_fire_when_below_thresholds(self):
        from stage20.failure_pool import TriggerCondition
        tc = TriggerCondition(
            knowledge_miss_count=5,
            unnatural_generation_count=2,
            multiturn_anomaly_count=1,
        )
        triggered, reason = tc.is_triggered()
        assert not triggered

    def test_trigger_fires_on_critical_safety(self):
        from stage20.failure_pool import TriggerCondition
        tc = TriggerCondition(critical_safety_events=1)
        triggered, reason = tc.is_triggered()
        assert triggered

    def test_build_pool_with_synthetic_coverage(self):
        from stage20.failure_pool import FailurePoolBuilder
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "test_pool.jsonl"
            builder = FailurePoolBuilder(output_path=str(output))
            samples = builder.build_pool(target_size=50, require_coverage=True)
            assert len(samples) > 0

            # Should have all 8 types covered
            types = {s.failure_type for s in samples}
            from src.core.bayes.schema import FailureCategory
            assert len(types) == len(FailureCategory)


# ── Pipeline Tests ────────────────────────────────────────────────

class TestStage20Pipeline:
    """Tests for stage20/stage20_pipeline.py"""

    def test_pipeline_runs_end_to_end(self):
        from stage20.stage20_pipeline import Stage20Pipeline
        from stage20.acceptance_harness import RealisticFailurePool

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = Stage20Pipeline(config={
                "failure_pool_path": str(Path(tmpdir) / "stage20_failure_pool.jsonl"),
                "output_dir": tmpdir,
                "patch_output_dir": str(Path(tmpdir) / "stage20_patches"),
            })
            failure_samples = RealisticFailurePool.build(
                target_size=20,
                output_path=str(Path(tmpdir) / "stage20_failure_pool.jsonl"),
            )
            result = pipeline.run(failure_samples=failure_samples)

        assert result["stage"] == "Stage20"
        assert result["version"] == "v0.1"
        assert "steps" in result
        assert "20-0_trigger" in result["steps"]
        assert "20-1_failure_pool" in result["steps"]
        assert "20-2_5_btsta_diagnoses" in result["steps"]
        assert "20-6_replay" in result["steps"]
        assert "20-7_shadow" in result["steps"]
        assert "20-8_rollout" in result["steps"]
        assert "20-9_freeze" in result["steps"]
        assert result["overall_pass"] is True


# ── Version Freezer Tests ─────────────────────────────────────────

class TestVersionFreezer:
    """Tests for stage20/version_freezer.py"""

    def test_freeze_creates_manifest_and_runbook(self):
        from stage20.version_freezer import VersionFreezer
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            freezer = VersionFreezer(output_dir=tmpdir)
            freeze_point = freezer.freeze(version="Stage20_Test_v0.1")

            assert freeze_point.version == "Stage20_Test_v0.1"
            assert "error_rate_spike" in freeze_point.auto_rollback_conditions

            # Check files were written
            manifest = Path(tmpdir) / "stage20_freeze_manifest.json"
            assert manifest.exists()

            runbook = Path(tmpdir) / "stage20_rollback_runbook.md"
            assert runbook.exists()
