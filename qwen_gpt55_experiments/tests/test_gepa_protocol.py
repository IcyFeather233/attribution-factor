import pytest

from qwen_gpt55_experiments.gepa.protocol import proposal_budget, require_completed_evolution


def test_budget_covers_seed_minibatches_and_validation():
    assert proposal_budget(22, 3, 4) == 134
    assert proposal_budget(22, 3, 1) == 50
    assert proposal_budget(1, 1, 1) == 4
    with pytest.raises(ValueError):
        proposal_budget(0, 3)
    with pytest.raises(ValueError):
        proposal_budget(22, 3, 0)


@pytest.mark.parametrize("summary", [{}, {"candidate_count": 1},
                                     {"optimization_completed": True, "new_candidates_evaluated": 0}])
def test_seed_only_and_legacy_summaries_cannot_resume(summary):
    with pytest.raises(RuntimeError, match="new candidate"):
        require_completed_evolution(summary)


@pytest.mark.parametrize("improves", [True, False])
def test_real_engine_evaluates_all_proposals_even_when_rejected(tmp_path, improves):
    from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything

    calls = []
    reflections = []

    def evaluator(candidate, example):
        calls.append((candidate, example))
        score = float(candidate != "seed") if improves else float(candidate == "seed")
        return score, {"Feedback": "Try a different instruction"}

    def reflect(prompt):
        reflections.append(prompt)
        return f"```\nimproved_{len(reflections)}\n```"

    result = optimize_anything(
        seed_candidate="seed", evaluator=evaluator, dataset=list(range(10)),
        valset=list(range(10, 32)),
        config=GEPAConfig(
            engine=EngineConfig(run_dir=str(tmp_path / "engine"),
                                max_metric_calls=proposal_budget(22, 3, 4),
                                max_candidate_proposals=4, max_workers=4, parallel=True,
                                acceptance_criterion="strict_improvement", write_agent_state=True),
            reflection=ReflectionConfig(reflection_lm=reflect, reflection_minibatch_size=3),
            merge=None, refiner=None,
        ),
    )
    assert len(reflections) == 4
    assert len({candidate for candidate, _ in calls if candidate != "seed"}) == 4
    assert (result.best_candidate != "seed") is improves
    require_completed_evolution({"optimization_completed": True, "new_candidates_evaluated": 1})
