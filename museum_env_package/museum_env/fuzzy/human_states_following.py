"""
human_states_following_v8.py
Scikit-fuzzy implementation of human_states_following_v8.fis.

Inputs:
    following_time : float, [0, 60]   seconds following the guide
    hhd            : float, [0, 4]    head-to-head distance (front)
    hrd            : float, [0, 4]    head-to-rear distance (back)
    density        : float, [0, 10]   crowd density

Outputs:
    overwhelmed : float, [0, 1]
    distracted  : float, [0, 1]
    impatient   : float, [0, 1]
    engaged     : float, [0, 1]

Usage:
    result = compute(following_time=30, hhd=0.25, hrd=0.5, density=7)
    print(result["dominant_state"])
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


# ------------------------------------------------------------------
# Universe of discourse
# ------------------------------------------------------------------
_RES = 1000
_DEFAULT = 0.2
_TIE_TOLERANCE = 0.01

following_time = ctrl.Antecedent(np.linspace(0, 60, _RES), "following_time")
hhd = ctrl.Antecedent(np.linspace(0, 4, _RES), "hhd")
hrd = ctrl.Antecedent(np.linspace(0, 4, _RES), "hrd")
density = ctrl.Antecedent(np.linspace(0, 10, _RES), "density")

engaged = ctrl.Consequent(np.linspace(0, 1, _RES), "engaged", defuzzify_method="centroid")
overwhelmed = ctrl.Consequent(np.linspace(0, 1, _RES), "overwhelmed", defuzzify_method="centroid")
distracted = ctrl.Consequent(np.linspace(0, 1, _RES), "distracted", defuzzify_method="centroid")
impatient = ctrl.Consequent(np.linspace(0, 1, _RES), "impatient", defuzzify_method="centroid")



# ------------------------------------------------------------------
# Input membership functions
# ------------------------------------------------------------------
following_time["short"] = fuzz.trapmf(following_time.universe, [0, 0, 10, 20])
following_time["medium"] = fuzz.trapmf(following_time.universe, [10, 20, 40, 50])
following_time["long"] = fuzz.trapmf(following_time.universe, [40, 50, 60, 60])

hhd["close"] = fuzz.trapmf(hhd.universe, [0, 0, 0.5, 0.6])
hhd["medium"] = fuzz.trapmf(hhd.universe, [0.5, 0.6, 0.9, 1.0])
hhd["far"] = fuzz.trapmf(hhd.universe, [0.9, 1.0, 4.0, 4.0])

hrd["close"] = fuzz.trapmf(hrd.universe, [0, 0, 1.0, 1.2])
hrd["medium"] = fuzz.trapmf(hrd.universe, [1.0, 1.2, 2.0, 2.2])
hrd["far"] = fuzz.trapmf(hrd.universe, [2.0, 2.2, 5.0, 5.0])

density["low"] = fuzz.trapmf(density.universe, [0, 0, 1, 1])
density["medium"] = fuzz.trapmf(density.universe, [2, 2, 4, 4])
density["crowded"] = fuzz.trapmf(density.universe, [5, 5, 10, 10])


# ------------------------------------------------------------------
# Output membership functions
# ------------------------------------------------------------------
for _var in [engaged, overwhelmed, distracted, impatient]:
    _var["low"] = fuzz.trapmf(_var.universe, [0, 0, 0.2, 0.4])
    _var["medium"] = fuzz.trimf(_var.universe, [0.3, 0.5, 0.7])
    _var["high"] = fuzz.trapmf(_var.universe, [0.6, 0.8, 1.0, 1.0])


# ------------------------------------------------------------------
# Rules
# FIS input order:  [following_time, hhd, hrd, density]
# FIS output order: [engaged, overwhelmed, distracted, impatient]
# ------------------------------------------------------------------
_mf = {
    "ft": {1: following_time["short"], 2: following_time["medium"], 3: following_time["long"]},
    "hhd": {1: hhd["close"], 2: hhd["medium"], 3: hhd["far"]},
    "hrd": {1: hrd["close"], 2: hrd["medium"], 3: hrd["far"]},
    "den": {1: density["low"], 2: density["medium"], 3: density["crowded"]},
    "eng": {1: engaged["low"], 2: engaged["medium"], 3: engaged["high"]},
    "ovw": {1: overwhelmed["low"], 2: overwhelmed["medium"], 3: overwhelmed["high"]},
    "dis": {1: distracted["low"], 2: distracted["medium"], 3: distracted["high"]},
    "imp": {1: impatient["low"], 2: impatient["medium"], 3: impatient["high"]},
}


def _make_rules(ft_i, hhd_i, hrd_i, den_i, eng_i, ovw_i, dis_i, imp_i):
    """
    Build scikit-fuzzy rules from the FIS integer encoding.

    The v8 FIS stores consequents in the order:
    [engaged, overwhelmed, distracted, impatient].
    """
    antecedents = []
    if ft_i:
        antecedents.append(_mf["ft"][ft_i])
    if hhd_i:
        antecedents.append(_mf["hhd"][hhd_i])
    if hrd_i:
        antecedents.append(_mf["hrd"][hrd_i])
    if den_i:
        antecedents.append(_mf["den"][den_i])
    if not antecedents:
        raise ValueError("Each fuzzy rule must contain at least one antecedent.")

    ant = antecedents[0]
    for antecedent in antecedents[1:]:
        ant = ant & antecedent

    rules_out = []
    if eng_i:
        rules_out.append(ctrl.Rule(ant, _mf["eng"][eng_i]))
    if ovw_i:
        rules_out.append(ctrl.Rule(ant, _mf["ovw"][ovw_i]))
    if dis_i:
        rules_out.append(ctrl.Rule(ant, _mf["dis"][dis_i]))
    if imp_i:
        rules_out.append(ctrl.Rule(ant, _mf["imp"][imp_i]))
    return rules_out


_RULE_TABLE = [
    (0, 1, 1, 3, 0, 3, 0, 0),  # close + close + crowded -> overwhelmed high
    (3, 3, 3, 1, 0, 0, 3, 0),  # long + far + far + low -> distracted high
    (3, 2, 3, 1, 0, 0, 3, 0),  # long + medium + far + low -> distracted high
    (3, 1, 1, 1, 0, 0, 0, 3),  # long + close + close + low -> impatient high
    (3, 1, 1, 2, 0, 0, 0, 3),  # long + close + close + medium -> impatient high
    (0, 0, 0, 1, 3, 0, 0, 0),  # density low -> engaged high
    (0, 0, 0, 2, 3, 0, 0, 0),  # density medium -> engaged high
    (0, 2, 0, 3, 2, 0, 0, 0),  # hhd medium + crowded -> engaged medium
    (0, 3, 0, 3, 2, 0, 0, 0),  # hhd far + crowded -> engaged medium
    (0, 1, 2, 3, 2, 0, 0, 0),  # hhd close + hrd medium + crowded -> engaged medium
    (0, 1, 3, 3, 2, 0, 0, 0),  # hhd close + hrd far + crowded -> engaged medium
    (3, 1, 1, 0, 1, 0, 0, 0),  # long + close + close -> engaged low
    (1, 0, 0, 0, 3, 0, 0, 0),  # short -> engaged high
    (3, 0, 3, 1, 1, 0, 0, 0),  # long + hrd far + low -> engaged low
]

rules = [rule for row in _RULE_TABLE for rule in _make_rules(*row)]


# ------------------------------------------------------------------
# Control system
# ------------------------------------------------------------------
_system = ctrl.ControlSystem(rules)
_simulation = ctrl.ControlSystemSimulation(_system)


def _select_dominant_state(results: dict[str, float], tie_tolerance: float = _TIE_TOLERANCE) -> str:
    """Pick the dominant state with engaged-first tie handling."""
    max_value = max(results.values())
    tied_states = [
        state for state, value in results.items() if max_value - float(value) <= tie_tolerance
    ]
    if "engaged" in tied_states:
        return "engaged"
    for state in ("overwhelmed", "distracted", "impatient", "engaged"):
        if state in tied_states:
            return state
    raise RuntimeError("Failed to resolve dominant state from fuzzy outputs.")


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------
def compute(following_time: float, hhd: float, hrd: float, density: float) -> dict:
    """
    Run the v8 fuzzy inference system for a single input vector.

    Returns a dict with keys:
    overwhelmed, distracted, impatient, engaged, dominant_state, dominant_value.

    If no rule fires for an output variable, that output defaults to 0.2.
    """
    _simulation.input["following_time"] = float(following_time)
    _simulation.input["hhd"] = float(hhd)
    _simulation.input["hrd"] = float(hrd)
    _simulation.input["density"] = float(density)
    _simulation.compute()

    results = {
        "overwhelmed": _simulation.output.get("overwhelmed", _DEFAULT),
        "distracted": _simulation.output.get("distracted", _DEFAULT),
        "impatient": _simulation.output.get("impatient", _DEFAULT),
        "engaged": _simulation.output.get("engaged", _DEFAULT),
    }

    dominant = _select_dominant_state(results)
    results["dominant_state"] = dominant
    results["dominant_value"] = results[dominant]
    return results


def compute_batch(inputs: np.ndarray) -> list[dict]:
    """Run inference on multiple input vectors."""
    return [compute(*row) for row in inputs]


# ------------------------------------------------------------------
# Quick self-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    import pandas as pd

    test_cases = [
        # (ft,   hhd,  hrd,  density, expected_dominant)
        (30, 0.25, 0.5, 7.0, "overwhelmed"),  # close + close + crowded
        (55, 2.0, 3.5, 0.5, "distracted"),  # long + far + far + low
        (55, 0.25, 0.5, 3.0, "impatient"),  # long + close + close + medium
        (5, 2.0, 3.5, 0.5, "engaged"),  # short + low fallback keeps engaged high
        (55, 0.25, 0.5, 0.5, "engaged"),  # engaged wins the close tie within tolerance
    ]

    print(
        f"{'ft':>5} {'hhd':>5} {'hrd':>5} {'den':>6}  "
        f"{'ovw':>6} {'dis':>6} {'imp':>6} {'eng':>6}  "
        f"{'dominant':<12} {'expected':<12} {'ok'}"
    )
    print("-" * 85)

    for ft_v, hhd_v, hrd_v, den_v, expected in test_cases:
        result = compute(ft_v, hhd_v, hrd_v, den_v)
        ok = "OK" if result["dominant_state"] == expected else "WARN"
        print(
            f"{ft_v:>5} {hhd_v:>5} {hrd_v:>5} {den_v:>6}  "
            f"{result['overwhelmed']:>6.3f} {result['distracted']:>6.3f} "
            f"{result['impatient']:>6.3f} {result['engaged']:>6.3f}  "
            f"{result['dominant_state']:<12} {expected:<12} {ok}"
        )

    print("\n=== 81-combination sweep ===")
    ft_vals = [5, 30, 55]
    hhd_vals = [0.25, 0.75, 2.0]
    hrd_vals = [0.5, 1.6, 3.5]
    den_vals = [0.5, 3.0, 7.0]
    ft_labels = ["short", "medium", "long"]
    hhd_labels = ["close", "medium", "far"]
    hrd_labels = ["close", "medium", "far"]
    den_labels = ["low", "medium", "crowded"]

    rows = []
    for i, fv in enumerate(ft_vals):
        for j, hv in enumerate(hhd_vals):
            for k, rv in enumerate(hrd_vals):
                for l, dv in enumerate(den_vals):
                    result = compute(fv, hv, rv, dv)
                    rows.append(
                        {
                            "following_time": ft_labels[i],
                            "hhd": hhd_labels[j],
                            "hrd": hrd_labels[k],
                            "density": den_labels[l],
                            **{
                                key: round(result[key], 4)
                                for key in ["overwhelmed", "distracted", "impatient", "engaged"]
                            },
                            "dominant_state": result["dominant_state"],
                            "dominant_value": round(result["dominant_value"], 4),
                        }
                    )

    df = pd.DataFrame(rows)
    print(df["dominant_state"].value_counts().to_string())
    print(f"\nTotal: {len(df)} combinations")
    df.to_csv("fis_81_combinations_python.csv", index=False)
    print("Saved -> fis_81_combinations_python.csv")
