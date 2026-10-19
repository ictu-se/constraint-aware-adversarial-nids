"""Regression checks for final budgets, immutable categories and benign rows."""
import numpy as np
import pandas as pd
import run_full_experiment as base
import run_q1_upgrade_experiment as repeated
from run_r2_checks import project


def main():
    train = pd.DataFrame({"count": [0., 1., 2., 3., 4., 5.],
                          "Protocol": [0., 6., 17., 6., 17., 6.],
                          "Dst Port": [80., 443., 53., 80., 443., 53.],
                          "continuous": [.12, .34, .56, .78, .91, .95]}, dtype=np.float32)
    constraints = base.infer_constraints(train, list(train.columns))
    original = train.iloc[[1, 3]].reset_index(drop=True)
    labels = np.array([0, 1])
    direction = np.ones(train.shape[1], dtype=np.float32)
    assert {1, 2}.issubset(set(constraints["immutable_idx"]))
    for epsilon in [.03, .06, .10, .15]:
        for attack in [base.make_adversarial, repeated.greedy_attack]:
            candidate, violation, *_ = attack(original, labels, direction, constraints, epsilon, True)
            assert violation == 0
            assert base.budget_violation_rate(candidate, original, constraints, epsilon) == 0
            assert np.array_equal(candidate.iloc[0], original.iloc[0])
            assert np.array_equal(candidate[["Protocol", "Dst Port"]], original[["Protocol", "Dst Port"]])
            # With span below 5, no positive integral increment fits epsilon<=.15.
            assert candidate.loc[1, "count"] == original.loc[1, "count"]
        x = original.to_numpy()
        query = project(x + 100, x, constraints, epsilon)
        assert base.budget_violation_rate(query, x, constraints, epsilon) == 0
        assert base.constraint_violations(query, x, constraints) == 0
        assert np.array_equal(query[:, [1, 2]], x[:, [1, 2]])
    # The original input remains admissible beyond training extrema.
    outside = original.copy()
    outside.loc[1, "count"] = 100.25
    candidate, violation, *_ = base.make_adversarial(outside, labels, direction, constraints, .03, True)
    assert violation == 0
    assert base.budget_violation_rate(candidate, outside, constraints, .03) == 0
    print("PASS: integer-budget regression, immutable protocol/ports, benign invariance, original anchors")


if __name__ == "__main__":
    main()
