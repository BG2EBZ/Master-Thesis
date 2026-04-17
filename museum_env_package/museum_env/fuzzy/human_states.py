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
    result = compute(following_time=30, hhd=0.25, hrd=0.5, density=7, context="listening")
    print(result["dominant_state"])
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


# ------------------------------------------------------------------
# Universe of discourse
# ------------------------------------------------------------------
_RES = 1000
_DEFAULT = 0.5
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
_INPUT_MF_PARAMS = {
    "following": {
        "following_time": {
            "short": [0, 0, 10, 20],
            "medium": [10, 20, 40, 50],
            "long": [40, 50, 60, 60],
        },
        "hhd": {
            "close": [0, 0, 0.5, 0.6],
            "medium": [0.5, 0.6, 0.9, 1.0],
            "far": [0.9, 1.0, 4.0, 4.0],
        },
        "hrd": {
            "close": [0, 0, 1.0, 1.2],
            "medium": [1.0, 1.2, 2.0, 2.2],
            "far": [2.0, 2.2, 5.0, 5.0],
        },
        "density": {
            "low": [0, 0, 1, 1],
            "medium": [2, 2, 4, 4],
            "crowded": [5, 5, 10, 10],
        },
    },
    "listening": {
        "following_time": {
            "short": [0, 0, 10, 20],
            "medium": [10, 20, 40, 50],
            "long": [40, 50, 60, 60],
        },
        "hhd": {
            "close": [0, 0, 0.5, 0.6],
            "medium": [0.5, 0.6, 0.9, 1.0],
            "far": [0.9, 1.0, 4.0, 4.0],
        },
        "hrd": {
            "close": [0, 0, 0.6, 0.8],
            "medium": [0.6, 0.8, 2.0, 2.2],
            "far": [2.0, 2.2, 5.0, 5.0],
        },
        "density": {
            "low": [0, 0, 2, 2],
            "medium": [3, 3, 7, 7],
            "crowded": [8, 8, 10, 10],
        },
    },
}

_CONTEXT_ALIASES = {
    "follow": "following",
    "following": "following",
    "listen": "listening",
    "listening": "listening",
}


def _normalize_context(context: str) -> str:
    normalized = _CONTEXT_ALIASES.get(str(context).strip().lower())
    if normalized is None:
        valid = ", ".join(sorted(_INPUT_MF_PARAMS))
        raise ValueError(f"Unknown fuzzy context: {context!r}. Expected one of: {valid}.")
    return normalized


def _create_variables():
    ft = ctrl.Antecedent(np.linspace(0, 60, _RES), "following_time")
    hhd_var = ctrl.Antecedent(np.linspace(0, 4, _RES), "hhd")
    hrd_var = ctrl.Antecedent(np.linspace(0, 4, _RES), "hrd")
    density_var = ctrl.Antecedent(np.linspace(0, 10, _RES), "density")

    engaged_var = ctrl.Consequent(np.linspace(0, 1, _RES), "engaged", defuzzify_method="centroid")
    overwhelmed_var = ctrl.Consequent(
        np.linspace(0, 1, _RES), "overwhelmed", defuzzify_method="centroid"
    )
    distracted_var = ctrl.Consequent(
        np.linspace(0, 1, _RES), "distracted", defuzzify_method="centroid"
    )
    impatient_var = ctrl.Consequent(np.linspace(0, 1, _RES), "impatient", defuzzify_method="centroid")
    return (
        ft,
        hhd_var,
        hrd_var,
        density_var,
        engaged_var,
        overwhelmed_var,
        distracted_var,
        impatient_var,
    )


def _build_membership_map(context: str):
    ft, hhd_var, hrd_var, density_var, engaged_var, overwhelmed_var, distracted_var, impatient_var = (
        _create_variables()
    )

    input_params = _INPUT_MF_PARAMS[_normalize_context(context)]
    antecedents = {
        "following_time": ft,
        "hhd": hhd_var,
        "hrd": hrd_var,
        "density": density_var,
    }
    for variable_name, variable in antecedents.items():
        for label, points in input_params[variable_name].items():
            variable[label] = fuzz.trapmf(variable.universe, points)

    for output_var in [engaged_var, overwhelmed_var, distracted_var, impatient_var]:
        output_var["low"] = fuzz.trapmf(output_var.universe, [0, 0, 0.2, 0.5])
        output_var["medium"] = fuzz.trimf(output_var.universe, [0.2, 0.5, 0.8])
        output_var["high"] = fuzz.trapmf(output_var.universe, [0.5, 0.8, 1.0, 1.0])

    return {
        "ft": {1: ft["short"], 2: ft["medium"], 3: ft["long"]},
        "hhd": {1: hhd_var["close"], 2: hhd_var["medium"], 3: hhd_var["far"]},
        "hrd": {1: hrd_var["close"], 2: hrd_var["medium"], 3: hrd_var["far"]},
        "den": {1: density_var["low"], 2: density_var["medium"], 3: density_var["crowded"]},
        "eng": {1: engaged_var["low"], 2: engaged_var["medium"], 3: engaged_var["high"]},
        "ovw": {1: overwhelmed_var["low"], 2: overwhelmed_var["medium"], 3: overwhelmed_var["high"]},
        "dis": {1: distracted_var["low"], 2: distracted_var["medium"], 3: distracted_var["high"]},
        "imp": {1: impatient_var["low"], 2: impatient_var["medium"], 3: impatient_var["high"]},
    }


# ------------------------------------------------------------------
# Output membership functions
# ------------------------------------------------------------------
# Output membership functions are shared across contexts and are built inside
# _build_membership_map() together with the selected input membership set.


# ------------------------------------------------------------------
# Rules
# FIS input order:  [following_time, hhd, hrd, density]
# FIS output order: [engaged, overwhelmed, distracted, impatient]
# ------------------------------------------------------------------
def _make_rules(membership_map, ft_i, hhd_i, hrd_i, den_i, eng_i, ovw_i, dis_i, imp_i):
    """
    Build scikit-fuzzy rules from the FIS integer encoding.

    The v8 FIS stores consequents in the order:
    [engaged, overwhelmed, distracted, impatient].
    """
    antecedents = []
    if ft_i:
        antecedents.append(membership_map["ft"][ft_i])
    if hhd_i:
        antecedents.append(membership_map["hhd"][hhd_i])
    if hrd_i:
        antecedents.append(membership_map["hrd"][hrd_i])
    if den_i:
        antecedents.append(membership_map["den"][den_i])
    if not antecedents:
        raise ValueError("Each fuzzy rule must contain at least one antecedent.")

    ant = antecedents[0]
    for antecedent in antecedents[1:]:
        ant = ant & antecedent

    rules_out = []
    if eng_i:
        rules_out.append(ctrl.Rule(ant, membership_map["eng"][eng_i]))
    if ovw_i:
        rules_out.append(ctrl.Rule(ant, membership_map["ovw"][ovw_i]))
    if dis_i:
        rules_out.append(ctrl.Rule(ant, membership_map["dis"][dis_i]))
    if imp_i:
        rules_out.append(ctrl.Rule(ant, membership_map["imp"][imp_i]))
    return rules_out


_RULE_TABLE = [
    (0, 1, 1, 3, 0, 3, 0, 0),  # hhd close + hrd close + crowded -> overwhelmed high
    (0, 3, 3, 1, 0, 0, 3, 0),  # hhd far + hrd far + low -> distracted high
    (0, 2, 3, 1, 0, 0, 3, 0),  # hhd medium + hrd far + low -> distracted high
    (0, 1, 1, 1, 0, 0, 0, 3),  # hhd close + hrd close + low -> impatient high
    (0, 1, 1, 2, 0, 0, 0, 3),  # hhd close + hrd close + medium -> impatient high
    (0, 0, 0, 1, 3, 0, 0, 0),  # density low -> engaged high
    (0, 0, 0, 2, 3, 0, 0, 0),  # density medium -> engaged high
    (0, 2, 0, 3, 2, 0, 0, 0),  # hhd medium + crowded -> engaged medium
    (0, 3, 0, 3, 2, 0, 0, 0),  # hhd far + crowded -> engaged medium
    (0, 1, 2, 3, 2, 0, 0, 0),  # hhd close + hrd medium + crowded -> engaged medium
    (0, 1, 3, 3, 2, 0, 0, 0),  # hhd close + hrd far + crowded -> engaged medium
    (3, 1, 1, 0, 1, 0, 0, 0),  # long + hhd close + hrd close -> engaged low
    (1, 0, 0, 0, 3, 0, 0, 0),  # short -> engaged high
    (3, 0, 3, 1, 1, 0, 0, 0),  # long + hrd far + low -> engaged low
    (0, 2, 0, 0, 3, 0, 0, 0),  # hhd medium -> engaged high
    (0, 0, 2, 0, 3, 0, 0, 0),  # hrd medium -> engaged high
]

def _build_control_system(context: str, rule_table=None):
    active_rule_table = _RULE_TABLE if rule_table is None else rule_table
    membership_map = _build_membership_map(context)
    rules = [rule for row in active_rule_table for rule in _make_rules(membership_map, *row)]
    return ctrl.ControlSystem(rules)


# ------------------------------------------------------------------
# Control system
# ------------------------------------------------------------------
_SYSTEMS = {context: _build_control_system(context) for context in _INPUT_MF_PARAMS}


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
def compute(
    following_time: float,
    hhd: float,
    hrd: float,
    density: float,
    context: str = "listening",
) -> dict:
    """
    Run the v8 fuzzy inference system for a single input vector.

    Returns a dict with keys:
    overwhelmed, distracted, impatient, engaged, dominant_state, dominant_value.

    If no rule fires for an output variable, that output defaults to _DEFAULT.
    """
    simulation = ctrl.ControlSystemSimulation(_SYSTEMS[_normalize_context(context)])
    simulation.input["following_time"] = float(following_time)
    simulation.input["hhd"] = float(hhd)
    simulation.input["hrd"] = float(hrd)
    simulation.input["density"] = float(density)
    simulation.compute()

    results = {
        "overwhelmed": simulation.output.get("overwhelmed", _DEFAULT),
        "distracted": simulation.output.get("distracted", _DEFAULT),
        "impatient": simulation.output.get("impatient", _DEFAULT),
        "engaged": simulation.output.get("engaged", _DEFAULT),
    }

    dominant = _select_dominant_state(results)
    results["dominant_state"] = dominant
    results["dominant_value"] = results[dominant]
    return results


def compute_following(following_time: float, hhd: float, hrd: float, density: float) -> dict:
    """Run inference with the following-stage input membership functions."""
    return compute(following_time=following_time, hhd=hhd, hrd=hrd, density=density, context="following")


def compute_listening(following_time: float, hhd: float, hrd: float, density: float) -> dict:
    """Run inference with the listening-stage input membership functions."""
    return compute(following_time=following_time, hhd=hhd, hrd=hrd, density=density, context="listening")


def compute_batch(inputs: np.ndarray, context: str = "listening") -> list[dict]:
    """Run inference on multiple input vectors."""
    return [compute(*row, context=context) for row in inputs]


# ------------------------------------------------------------------
# Quick self-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    import pandas as pd

    legacy_rule_table = [
        (0, 1, 1, 3, 0, 3, 0, 0),
        (3, 3, 3, 1, 0, 0, 3, 0),
        (3, 2, 3, 1, 0, 0, 3, 0),
        (3, 1, 1, 1, 0, 0, 0, 3),
        (3, 1, 1, 2, 0, 0, 0, 3),
        (0, 0, 0, 1, 3, 0, 0, 0),
        (0, 0, 0, 2, 3, 0, 0, 0),
        (0, 2, 0, 3, 2, 0, 0, 0),
        (0, 3, 0, 3, 2, 0, 0, 0),
        (0, 1, 2, 3, 2, 0, 0, 0),
        (0, 1, 3, 3, 2, 0, 0, 0),
        (3, 1, 1, 0, 1, 0, 0, 0),
        (1, 0, 0, 0, 3, 0, 0, 0),
        (3, 0, 3, 1, 1, 0, 0, 0),
        (0, 2, 0, 0, 3, 0, 0, 0),
        (0, 0, 2, 0, 3, 0, 0, 0),
    ]

    def compute_with_rule_table(
        rule_table,
        following_time_value,
        hhd_value,
        hrd_value,
        density_value,
        context="listening",
    ):
        local_system = _build_control_system(context=context, rule_table=rule_table)
        local_simulation = ctrl.ControlSystemSimulation(local_system)
        local_simulation.input["following_time"] = float(following_time_value)
        local_simulation.input["hhd"] = float(hhd_value)
        local_simulation.input["hrd"] = float(hrd_value)
        local_simulation.input["density"] = float(density_value)
        local_simulation.compute()

        local_results = {
            "overwhelmed": local_simulation.output.get("overwhelmed", _DEFAULT),
            "distracted": local_simulation.output.get("distracted", _DEFAULT),
            "impatient": local_simulation.output.get("impatient", _DEFAULT),
            "engaged": local_simulation.output.get("engaged", _DEFAULT),
        }
        dominant_state = _select_dominant_state(local_results)
        local_results["dominant_state"] = dominant_state
        local_results["dominant_value"] = local_results[dominant_state]
        return local_results

    smoke_test_cases = [
        # Experimental-rule smoke tests with fixed dominant-state expectations.
        # Long-duration distracted/impatient cases are kept to confirm the original hits still work.
        (30, 0.25, 0.5, 8.0, "overwhelmed"),  # close + close + crowded
        (55, 2.0, 3.5, 0.5, "distracted"),  # long + far + far + low
        (55, 0.25, 0.5, 3.0, "impatient"),  # long + close + close + medium
        (5, 2.0, 3.5, 0.5, "engaged"),  # engaged-first tie still wins at short time
        (55, 0.25, 0.5, 0.5, "impatient"),  # low-density impatient remains dominant
    ]

    diagnostic_cases = [
        # Experimental diagnostics: remove the time gate and confirm these states now participate
        # even at medium following time, without requiring them to become dominant.
        ("distracted", 30, 2.0, 3.5, 0.5),
        ("impatient", 30, 0.25, 0.5, 3.0),
    ]

    print(
        f"{'ft':>5} {'hhd':>5} {'hrd':>5} {'den':>6}  "
        f"{'ovw':>6} {'dis':>6} {'imp':>6} {'eng':>6}  "
        f"{'dominant':<12} {'expected':<12} {'ok'}"
    )
    print("-" * 85)

    for ft_v, hhd_v, hrd_v, den_v, expected in smoke_test_cases:
        result = compute(ft_v, hhd_v, hrd_v, den_v)
        ok = "OK" if result["dominant_state"] == expected else "WARN"
        print(
            f"{ft_v:>5} {hhd_v:>5} {hrd_v:>5} {den_v:>6}  "
            f"{result['overwhelmed']:>6.3f} {result['distracted']:>6.3f} "
            f"{result['impatient']:>6.3f} {result['engaged']:>6.3f}  "
            f"{result['dominant_state']:<12} {expected:<12} {ok}"
        )

    print("\n=== Time-Gate Diagnostics vs Legacy Rules ===")
    print(
        f"{'state':<11} {'ft':>5} {'hhd':>5} {'hrd':>5} {'den':>6}  "
        f"{'legacy':>8} {'current':>8} {'delta':>8} {'dominant':<12} {'ok'}"
    )
    print("-" * 95)

    for target_state, ft_v, hhd_v, hrd_v, den_v in diagnostic_cases:
        legacy_result = compute_with_rule_table(
            legacy_rule_table,
            ft_v,
            hhd_v,
            hrd_v,
            den_v,
        )
        current_result = compute(ft_v, hhd_v, hrd_v, den_v)
        delta = float(current_result[target_state]) - float(legacy_result[target_state])
        ok = "OK" if delta > 1e-6 else "WARN"
        print(
            f"{target_state:<11} {ft_v:>5} {hhd_v:>5} {hrd_v:>5} {den_v:>6}  "
            f"{legacy_result[target_state]:>8.3f} {current_result[target_state]:>8.3f} "
            f"{delta:>8.3f} {current_result['dominant_state']:<12} {ok}"
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
