"""
Scikit-fuzzy human-state inference for following and listening contexts.

Inputs:
    following_time : float, [0, 60]   seconds in the active phase
    hhd            : float, [0, 4]    head-to-head distance (front)
    hrd            : float, [0, 4]    head-to-rear distance (back)
    density        : float, [0, 10]   crowd density
    angle          : float, [-180, 180] robot-relative bearing in degrees

Outputs:
    overwhelmed : float, [0, 1]
    distracted  : float, [0, 1]
    impatient   : float, [0, 1]
    engaged     : float, [0, 1]

Usage:
    result = compute(
        following_time=30,
        hhd=0.25,
        hrd=0.5,
        density=7,
        angle=0,
        context="listening",
    )
    print(result["dominant_state"])
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from ..human import HumanProfile


_RES = 1000
_DEFAULT = 0.5
_TIE_TOLERANCE = 0.01
_CONTEXTS = ("following", "listening")
_PROFILES = (HumanProfile.NORMAL, HumanProfile.NEURODIVERGENT)

_CONTEXT_ALIASES = {
    "follow": "following",
    "following": "following",
    "listen": "listening",
    "listening": "listening",
}


def _normalize_context(context: str) -> str:
    normalized = _CONTEXT_ALIASES.get(str(context).strip().lower())
    if normalized is None:
        valid = ", ".join(_CONTEXTS)
        raise ValueError(f"Unknown fuzzy context: {context!r}. Expected one of: {valid}.")
    return normalized


def _define_normal_input_mfs(context: str, ft, hhd, hrd, density, angle) -> None:
    if context == "following":
        ft["short"] = fuzz.trapmf(ft.universe, [0, 0, 15, 25])
        ft["medium"] = fuzz.trapmf(ft.universe, [15, 25, 35, 45])
        ft["long"] = fuzz.trapmf(ft.universe, [35, 45, 60, 60])

        hhd["close"] = fuzz.trapmf(hhd.universe, [0, 0, 0.5, 0.7])
        hhd["medium"] = fuzz.trapmf(hhd.universe, [0.5, 0.7, 1.0, 1.2])
        hhd["far"] = fuzz.trapmf(hhd.universe, [1.0, 1.2, 4.0, 4.0])

        hrd["close"] = fuzz.trapmf(hrd.universe, [0, 0, 0.8, 1.0])
        hrd["medium"] = fuzz.trapmf(hrd.universe, [0.8, 1.0, 2.0, 2.2])
        hrd["far"] = fuzz.trapmf(hrd.universe, [2.0, 2.2, 5.0, 5.0])

        density["low"] = fuzz.trapmf(density.universe, [0, 0, 3, 3])
        density["medium"] = fuzz.trapmf(density.universe, [4, 4, 7, 7])
        density["crowded"] = fuzz.trapmf(density.universe, [8, 8, 10, 10])

        angle["ahead"] = fuzz.trapmf(angle.universe, [-35, -15, 15, 35])
        angle["left"] = fuzz.trapmf(angle.universe, [-110, -90, -35, -15])
        angle["right"] = fuzz.trapmf(angle.universe, [15, 35, 90, 110])
        angle["behind_left"] = fuzz.trapmf(angle.universe, [-180, -180, -110, -90])
        angle["behind_right"] = fuzz.trapmf(angle.universe, [90, 110, 180, 180])
        return

    ft["short"] = fuzz.trapmf(ft.universe, [0, 0, 5, 10])
    ft["medium"] = fuzz.trapmf(ft.universe, [10, 15, 20, 25])
    ft["long"] = fuzz.trapmf(ft.universe, [20, 25, 60, 60])

    hhd["close"] = fuzz.trapmf(hhd.universe, [0, 0, 0.5, 0.6])
    hhd["medium"] = fuzz.trapmf(hhd.universe, [0.5, 0.6, 0.9, 1.0])
    hhd["far"] = fuzz.trapmf(hhd.universe, [0.9, 1.0, 4.0, 4.0])

    hrd["close"] = fuzz.trapmf(hrd.universe, [0, 0, 0.6, 0.8])
    hrd["medium"] = fuzz.trapmf(hrd.universe, [0.6, 0.8, 2.0, 2.2])
    hrd["far"] = fuzz.trapmf(hrd.universe, [2.0, 2.2, 5.0, 5.0])

    density["low"] = fuzz.trapmf(density.universe, [0, 0, 3, 3])
    density["medium"] = fuzz.trapmf(density.universe, [4, 4, 7, 7])
    density["crowded"] = fuzz.trapmf(density.universe, [8, 8, 10, 10])

    angle["ahead"] = fuzz.trapmf(angle.universe, [-35, -15, 15, 35])
    angle["left"] = fuzz.trapmf(angle.universe, [-110, -90, -35, -15])
    angle["right"] = fuzz.trapmf(angle.universe, [15, 35, 90, 110])
    angle["behind_left"] = fuzz.trapmf(angle.universe, [-180, -180, -110, -90])
    angle["behind_right"] = fuzz.trapmf(angle.universe, [90, 110, 180, 180])

def _define_nd_input_mfs(context: str, ft, hhd, hrd, density, angle) -> None:
    if context == "following":
        ft["short"] = fuzz.trapmf(ft.universe, [0, 0, 8, 15])
        ft["medium"] = fuzz.trapmf(ft.universe, [8, 15, 25, 35])
        ft["long"] = fuzz.trapmf(ft.universe, [25, 35, 60, 60])

        hhd["close"] = fuzz.trapmf(hhd.universe, [0, 0, 0.7, 1.0])
        hhd["medium"] = fuzz.trapmf(hhd.universe, [0.7, 1.0, 1.3, 1.6])
        hhd["far"] = fuzz.trapmf(hhd.universe, [1.3, 1.6, 4.0, 4.0])

        hrd["close"] = fuzz.trapmf(hrd.universe, [0, 0, 1.0, 1.3])
        hrd["medium"] = fuzz.trapmf(hrd.universe, [1.0, 1.3, 2.3, 2.5])
        hrd["far"] = fuzz.trapmf(hrd.universe, [2.3, 2.5, 5.0, 5.0])

        density["low"] = fuzz.trapmf(density.universe, [0, 0, 2, 2])
        density["medium"] = fuzz.trapmf(density.universe, [3, 3, 5, 5])
        density["crowded"] = fuzz.trapmf(density.universe, [6, 6, 10, 10])

        angle["ahead"] = fuzz.trapmf(angle.universe, [-35, -15, 15, 35])
        angle["left"] = fuzz.trapmf(angle.universe, [-110, -90, -35, -15])
        angle["right"] = fuzz.trapmf(angle.universe, [15, 35, 90, 110])
        angle["behind_left"] = fuzz.trapmf(angle.universe, [-180, -180, -110, -90])
        angle["behind_right"] = fuzz.trapmf(angle.universe, [90, 110, 180, 180])
        return

    ft["short"] = fuzz.trapmf(ft.universe, [0, 0, 4, 8])
    ft["medium"] = fuzz.trapmf(ft.universe, [8, 12, 18, 22])
    ft["long"] = fuzz.trapmf(ft.universe, [18, 22, 60, 60])

    hhd["close"] = fuzz.trapmf(hhd.universe, [0, 0, 0.8, 1.0])
    hhd["medium"] = fuzz.trapmf(hhd.universe, [0.8, 1.0, 1.2, 1.4])
    hhd["far"] = fuzz.trapmf(hhd.universe, [1.2, 1.4, 4.0, 4.0])

    hrd["close"] = fuzz.trapmf(hrd.universe, [0, 0, 0.9, 1.1])
    hrd["medium"] = fuzz.trapmf(hrd.universe, [0.9, 1.1, 2.2, 2.4])
    hrd["far"] = fuzz.trapmf(hrd.universe, [2.2, 2.4, 5.0, 5.0])

    density["low"] = fuzz.trapmf(density.universe, [0, 0, 2, 2])
    density["medium"] = fuzz.trapmf(density.universe, [3, 3, 5, 5])
    density["crowded"] = fuzz.trapmf(density.universe, [6, 6, 10, 10])

    angle["ahead"] = fuzz.trapmf(angle.universe, [-35, -15, 15, 35])
    angle["left"] = fuzz.trapmf(angle.universe, [-110, -90, -35, -15])
    angle["right"] = fuzz.trapmf(angle.universe, [15, 35, 90, 110])
    angle["behind_left"] = fuzz.trapmf(angle.universe, [-180, -180, -110, -90])
    angle["behind_right"] = fuzz.trapmf(angle.universe, [90, 110, 180, 180])

def _define_input_mfs(context: str, profile: str, ft, hhd, hrd, density, angle) -> None:
    if profile == HumanProfile.NEURODIVERGENT:
        _define_nd_input_mfs(context, ft, hhd, hrd, density, angle)
        return
    _define_normal_input_mfs(context, ft, hhd, hrd, density, angle)


def _define_output_mfs(engaged, overwhelmed, distracted, impatient) -> None:
    for output_var in (engaged, overwhelmed, distracted, impatient):
        output_var["low"] = fuzz.trapmf(output_var.universe, [0, 0, 0.2, 0.5])
        output_var["medium"] = fuzz.trimf(output_var.universe, [0.2, 0.5, 0.8])
        output_var["high"] = fuzz.trapmf(output_var.universe, [0.5, 0.8, 1.0, 1.0])


def _build_rules(ft, hhd, hrd, density, angle, engaged, overwhelmed, distracted, impatient, curiosity) -> list[ctrl.Rule]:
    return [
        # Overwhelmed rules
        ctrl.Rule(hhd["close"] & hrd["close"] & density["crowded"], overwhelmed["high"]),

        # Distracted rules
        ctrl.Rule(hhd["far"] & hrd["far"] & density["low"], distracted["high"]),
        ctrl.Rule(hhd["medium"] & hrd["far"] & density["low"], distracted["high"]),
        ctrl.Rule(hhd["far"] & hrd["medium"] & density["low"], distracted["high"]),
        ctrl.Rule(hhd["medium"] & hrd["medium"] & density["low"], distracted["high"]),

        # Impatient rulesW
        ctrl.Rule(hhd["close"] & hrd["close"] & density["low"], impatient["high"]),
        ctrl.Rule(hhd["close"] & hrd["close"] & density["medium"], impatient["high"]),

        # Curiosity rules
        ctrl.Rule(hrd["close"] & angle["ahead"], curiosity["high"]),
        ctrl.Rule(hrd["medium"] & angle["ahead"], curiosity["high"]),
                   
        # Engaged rules
        ctrl.Rule(density["low"], engaged["high"]),
        ctrl.Rule(density["medium"], engaged["high"]),
        ctrl.Rule(hhd["medium"] & density["crowded"], engaged["medium"]),
        ctrl.Rule(hhd["far"] & density["crowded"], engaged["medium"]),
        ctrl.Rule(hhd["close"] & hrd["medium"] & density["crowded"], engaged["medium"]),
        ctrl.Rule(hhd["close"] & hrd["far"] & density["crowded"], engaged["medium"]),
        ctrl.Rule(ft["long"] & hhd["close"] & hrd["close"], engaged["low"]),
        ctrl.Rule(ft["short"], engaged["high"]),
        ctrl.Rule(ft["long"] & hrd["far"] & density["low"], engaged["low"]),
        ctrl.Rule(hhd["medium"], engaged["high"]),
        ctrl.Rule(hrd["medium"], engaged["high"]),
    ]


def _build_system(context: str, profile: str) -> ctrl.ControlSystem:
    normalized_context = _normalize_context(context)
    normalized_profile = str(profile).strip().lower()

    ft = ctrl.Antecedent(np.linspace(0, 60, _RES), "following_time")
    hhd = ctrl.Antecedent(np.linspace(0, 4, _RES), "hhd")
    hrd = ctrl.Antecedent(np.linspace(0, 4, _RES), "hrd")
    density = ctrl.Antecedent(np.linspace(0, 10, _RES), "density")
    angle = ctrl.Antecedent(np.linspace(-180, 180, _RES), "angle")

    engaged = ctrl.Consequent(np.linspace(0, 1, _RES), "engaged", defuzzify_method="centroid")
    overwhelmed = ctrl.Consequent(np.linspace(0, 1, _RES), "overwhelmed", defuzzify_method="centroid")
    distracted = ctrl.Consequent(np.linspace(0, 1, _RES), "distracted", defuzzify_method="centroid")
    impatient = ctrl.Consequent(np.linspace(0, 1, _RES), "impatient", defuzzify_method="centroid")

    _define_input_mfs(normalized_context, normalized_profile, ft, hhd, hrd, density, angle)
    _define_output_mfs(engaged, overwhelmed, distracted, impatient)
    rules = _build_rules(ft, hhd, hrd, density, angle, engaged, overwhelmed, distracted, impatient)
    return ctrl.ControlSystem(rules)


_SYSTEMS = {
    ("following", "normal"): _build_system("following", "normal"),
    ("following", "neurodivergent"): _build_system("following", "neurodivergent"),
    ("listening", "normal"): _build_system("listening", "normal"),
    ("listening", "neurodivergent"): _build_system("listening", "neurodivergent"),
}


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


def compute(
    following_time: float,
    hhd: float,
    hrd: float,
    density: float,
    angle: float,
    *,
    context: str,
    profile: str = HumanProfile.NORMAL,
) -> dict:
    """
    Run the fuzzy inference system for a single input vector.

    Returns a dict with keys:
    overwhelmed, distracted, impatient, engaged, dominant_state, dominant_value.
    """
    normalized_context = _normalize_context(context)
    normalized_profile = str(profile).strip().lower()
    if normalized_profile not in _PROFILES:
        valid = ", ".join(_PROFILES)
        raise ValueError(f"Unknown fuzzy profile: {profile!r}. Expected one of: {valid}.")
    simulation = ctrl.ControlSystemSimulation(_SYSTEMS[(normalized_context, normalized_profile)])
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


def compute_batch(
    inputs: np.ndarray,
    *,
    context: str,
    profile: str = HumanProfile.NORMAL,
) -> list[dict]:
    """Run inference on multiple input vectors."""
    rows = np.asarray(inputs, dtype=np.float32)
    return [compute(*row, context=context, profile=profile) for row in rows]


if __name__ == "__main__":
    sample_input = (3.1, 0.56, 1.05, 5.0, 0.0)

    for sample_context in _CONTEXTS:
        result = compute(*sample_input, context=sample_context)
        print(
            f"{sample_context:<10} "
            f"dominant={result['dominant_state']:<12} "
            f"value={result['dominant_value']:.3f}"
        )
