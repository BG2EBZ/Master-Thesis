"""
Fast human-state fuzzy inference for following and listening contexts.

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
    curiosity   : float, [0, 1]
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ..human import HumanProfile


_RES = 1000
_DEFAULT = 0.5
_CURIOSITY_DEFAULT = 0.0
_TIE_TOLERANCE = 0.01
_CONTEXTS = ("following", "listening")
_PROFILES = (HumanProfile.NORMAL, HumanProfile.NEURODIVERGENT)
_OUTPUT_NAMES = ("overwhelmed", "distracted", "impatient", "engaged", "curiosity")
_OUTPUT_TERMS = ("low", "medium", "high")
_OUTPUT_DEFAULTS = {
    "overwhelmed": _DEFAULT,
    "distracted": _DEFAULT,
    "impatient": _DEFAULT,
    "engaged": _DEFAULT,
    "curiosity": _CURIOSITY_DEFAULT,
}
_OUTPUT_UNIVERSE = np.linspace(0.0, 1.0, _RES, dtype=np.float64)
_INPUT_UNIVERSES = {
    "following_time": np.linspace(0.0, 60.0, _RES, dtype=np.float64),
    "hhd": np.linspace(0.0, 4.0, _RES, dtype=np.float64),
    "hrd": np.linspace(0.0, 4.0, _RES, dtype=np.float64),
    "density": np.linspace(0.0, 10.0, _RES, dtype=np.float64),
    "angle": np.linspace(-180.0, 180.0, _RES, dtype=np.float64),
}

_CONTEXT_ALIASES = {
    "follow": "following",
    "following": "following",
    "listen": "listening",
    "listening": "listening",
}


@dataclass(frozen=True)
class MFSpec:
    shape: str
    points: tuple[float, ...]


@dataclass(frozen=True)
class RuleSpec:
    antecedents: tuple[tuple[str, str], ...]
    consequent_name: str
    consequent_term: str


@dataclass(frozen=True)
class FuzzySystemDefinition:
    input_specs: dict[str, dict[str, MFSpec]]
    input_curves: dict[str, dict[str, np.ndarray]]
    rules: tuple[RuleSpec, ...]
    output_curves: dict[str, np.ndarray]


def _trap(a: float, b: float, c: float, d: float) -> MFSpec:
    return MFSpec("trap", (float(a), float(b), float(c), float(d)))


def _tri(a: float, b: float, c: float) -> MFSpec:
    return MFSpec("tri", (float(a), float(b), float(c)))


def _normalize_context(context: str) -> str:
    normalized = _CONTEXT_ALIASES.get(str(context).strip().lower())
    if normalized is None:
        valid = ", ".join(_CONTEXTS)
        raise ValueError(f"Unknown fuzzy context: {context!r}. Expected one of: {valid}.")
    return normalized


def _normalize_profile(profile: str) -> str:
    normalized_profile = str(profile).strip().lower()
    if normalized_profile not in _PROFILES:
        valid = ", ".join(_PROFILES)
        raise ValueError(f"Unknown fuzzy profile: {profile!r}. Expected one of: {valid}.")
    return normalized_profile


def _trapmf(values, spec: MFSpec) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    a, b, c, d = spec.points
    result = np.zeros_like(x, dtype=np.float64)

    plateau_mask = (x >= b) & (x <= c)
    result[plateau_mask] = 1.0

    if a != b:
        left_mask = (x > a) & (x < b)
        result[left_mask] = (x[left_mask] - a) / (b - a)

    if c != d:
        right_mask = (x > c) & (x < d)
        result[right_mask] = (d - x[right_mask]) / (d - c)

    return np.clip(result, 0.0, 1.0)


def _trimf(values, spec: MFSpec) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    a, b, c = spec.points
    result = np.zeros_like(x, dtype=np.float64)

    if a != b:
        left_mask = (x > a) & (x < b)
        result[left_mask] = (x[left_mask] - a) / (b - a)
    if b != c:
        right_mask = (x > b) & (x < c)
        result[right_mask] = (c - x[right_mask]) / (c - b)
    result[x == b] = 1.0

    return np.clip(result, 0.0, 1.0)


def _mf(values, spec: MFSpec) -> np.ndarray:
    if spec.shape == "trap":
        return _trapmf(values, spec)
    if spec.shape == "tri":
        return _trimf(values, spec)
    raise ValueError(f"Unsupported membership-function shape: {spec.shape!r}")


def _build_input_specs(context: str, profile: str) -> dict[str, dict[str, MFSpec]]:
    if profile == HumanProfile.NEURODIVERGENT:
        if context == "following":
            return {
                "following_time": {
                    "short": _trap(0, 0, 8, 15),
                    "medium": _trap(8, 15, 30, 35),
                    "long": _trap(30, 35, 60, 60),
                },
                "hhd": {
                    "close": _trap(0, 0, 0.7, 1.0),
                    "medium": _trap(0.7, 1.0, 1.3, 1.6),
                    "far": _trap(1.3, 1.6, 4.0, 4.0),
                },
                "hrd": {
                    "close": _trap(0, 0, 1.0, 1.3),
                    "medium": _trap(1.0, 1.3, 2.3, 2.5),
                    "far": _trap(2.3, 2.5, 5.0, 5.0),
                },
                "density": {
                    "low": _trap(0, 0, 2, 2),
                    "medium": _trap(3, 3, 6, 6),
                    "crowded": _trap(7, 7, 10, 10),
                },
                "angle": {
                    "ahead": _trap(-35, -15, 15, 35),
                },
            }
        return {
            "following_time": {
                "short": _trap(0, 0, 4, 8),
                "medium": _trap(8, 12, 18, 22),
                "long": _trap(18, 22, 60, 60),
            },
            "hhd": {
                "close": _trap(0, 0, 0.8, 1.0),
                "medium": _trap(0.8, 1.0, 1.2, 1.4),
                "far": _trap(1.2, 1.4, 4.0, 4.0),
            },
            "hrd": {
                "close": _trap(0, 0, 0.9, 1.1),
                "medium": _trap(0.9, 1.1, 2.2, 2.4),
                "far": _trap(2.2, 2.4, 5.0, 5.0),
            },
            "density": {
                "low": _trap(0, 0, 2, 2),
                "medium": _trap(3, 3, 6, 6),
                "crowded": _trap(7, 7, 10, 10),
            },
            "angle": {
                "ahead": _trap(-45, -20, 20, 45),
            },
        }

    if context == "following":
        return {
            "following_time": {
                "short": _trap(0, 0, 20, 30),
                "medium": _trap(20, 30, 40, 50),
                "long": _trap(40, 50, 120, 120),
            },
            "hhd": {
                "close": _trap(0, 0, 0.5, 0.7),
                "medium": _trap(0.5, 0.7, 1.3, 1.5),
                "far": _trap(1.3, 1.5, 4.0, 4.0),
            },
            "hrd": {
                "close": _trap(0, 0, 0.8, 1.0),
                "medium": _trap(0.8, 1.0, 2.0, 2.2),
                "far": _trap(2.0, 2.2, 5.0, 5.0),
            },
            "density": {
                "low": _trap(0, 0, 3, 3),
                "medium": _trap(4, 4, 8, 8),
                "crowded": _trap(9, 9, 12, 12),
            },
            "angle": {
                "ahead": _trap(-45, -30, 30, 45),
            },
        }

    return {
        "following_time": {
            "short": _trap(0, 0, 5, 10),
            "medium": _trap(10, 15, 20, 25),
            "long": _trap(20, 25, 60, 60),
        },
        "hhd": {
            "close": _trap(0, 0, 0.5, 0.6),
            "medium": _trap(0.5, 0.6, 0.9, 1.0),
            "far": _trap(0.9, 1.0, 4.0, 4.0),
        },
        "hrd": {
            "close": _trap(0, 0, 0.6, 0.8),
            "medium": _trap(0.6, 0.8, 2.0, 2.2),
            "far": _trap(2.0, 2.2, 5.0, 5.0),
        },
        "density": {
            "low": _trap(0, 0, 3, 3),
            "medium": _trap(4, 4, 8, 8),
            "crowded": _trap(9, 9, 12, 12),
        },
        "angle": {
            "ahead": _trap(-45, -30, 30, 45),
        },
    }


def _build_output_curves() -> dict[str, np.ndarray]:
    term_specs = {
        "low": _trap(0, 0, 0.2, 0.5),
        "medium": _tri(0.2, 0.5, 0.8),
        "high": _trap(0.5, 0.8, 1.0, 1.0),
    }
    return {
        term: np.asarray(_mf(_OUTPUT_UNIVERSE, spec), dtype=np.float64)
        for term, spec in term_specs.items()
    }


def _build_input_curves(input_specs: dict[str, dict[str, MFSpec]]) -> dict[str, dict[str, np.ndarray]]:
    return {
        var_name: {
            term_name: np.asarray(_mf(_INPUT_UNIVERSES[var_name], spec), dtype=np.float64)
            for term_name, spec in term_specs.items()
        }
        for var_name, term_specs in input_specs.items()
    }


def _build_rules() -> tuple[RuleSpec, ...]:
    return (
        RuleSpec((("following_time", "long"), ("hhd", "close"), ("hrd", "close"), ("density", "crowded")), "overwhelmed", "high"),
        RuleSpec((("following_time", "medium"), ("hhd", "close"), ("hrd", "close"), ("density", "crowded")), "overwhelmed", "high"),

        # Spatial disengagement: visitor is physically/socially detached.
        RuleSpec((("hhd", "far"), ("hrd", "far"), ("density", "low")), "distracted", "high"),
        RuleSpec((("hhd", "medium"), ("hrd", "far"), ("density", "low")), "distracted", "medium"),
        RuleSpec((("hhd", "far"), ("hrd", "medium"), ("density", "low")), "distracted", "medium"),

        # Time-enhanced distraction: long explanation/following increases attention decay.
        RuleSpec((("following_time", "long"), ("hhd", "far"), ("hrd", "far"), ("density", "low")), "distracted", "high"),
        RuleSpec((("following_time", "long"), ("hhd", "medium"), ("hrd", "far"), ("density", "low")), "distracted", "high"),
        RuleSpec((("following_time", "long"), ("hhd", "far"), ("hrd", "medium"), ("density", "low")), "distracted", "high"),

        RuleSpec((("following_time", "long"), ("hhd", "close"), ("hrd", "close"), ("density", "low")), "impatient", "high"),
        RuleSpec((("following_time", "long"), ("hhd", "close"), ("hrd", "close"), ("density", "medium")), "impatient", "high"),
        RuleSpec((("following_time", "medium"), ("hhd", "close"), ("hrd", "close"), ("density", "low")), "impatient", "high"),
        RuleSpec((("following_time", "medium"), ("hhd", "close"), ("hrd", "close"), ("density", "medium")), "impatient", "high"),

        RuleSpec((("hrd", "close"), ("angle", "ahead")), "curiosity", "high"),

        RuleSpec((("density", "low"),), "engaged", "high"),
        RuleSpec((("density", "medium"),), "engaged", "high"),
        RuleSpec((("hhd", "medium"), ("density", "crowded")), "engaged", "medium"),
        RuleSpec((("hhd", "far"), ("density", "crowded")), "engaged", "medium"),
        RuleSpec((("hhd", "close"), ("hrd", "medium"), ("density", "crowded")), "engaged", "medium"),
        RuleSpec((("hhd", "close"), ("hrd", "far"), ("density", "crowded")), "engaged", "medium"),
        RuleSpec((("following_time", "long"), ("hhd", "close"), ("hrd", "close")), "engaged", "low"),
        RuleSpec((("following_time", "short"),), "engaged", "high"),
        RuleSpec((("following_time", "long"), ("hrd", "far"), ("density", "low")), "engaged", "low"),
        RuleSpec((("hhd", "medium"),), "engaged", "high"),
        RuleSpec((("hrd", "medium"),), "engaged", "high"),
        RuleSpec((("hrd", "close"), ("angle", "ahead")), "engaged", "medium"),
    )


_SHARED_OUTPUT_CURVES = _build_output_curves()
_SHARED_RULES = _build_rules()
_SYSTEMS = {
    (context, profile): FuzzySystemDefinition(
        input_specs=_build_input_specs(context, profile),
        input_curves=_build_input_curves(_build_input_specs(context, profile)),
        rules=_SHARED_RULES,
        output_curves=_SHARED_OUTPUT_CURVES,
    )
    for context in _CONTEXTS
    for profile in _PROFILES
}


def in_ahead_region(angle_value: float, *, context: str, profile: str = HumanProfile.NORMAL) -> bool:
    """Return whether the angle falls inside the fuzzy `ahead` support."""
    normalized_context = _normalize_context(context)
    _normalize_profile(profile)

    lower_deg, upper_deg = (-35.0, 35.0) if normalized_context == "following" else (-45.0, 45.0)
    angle_deg = float(angle_value)
    return bool(lower_deg < angle_deg < upper_deg)


def _select_dominant_state(results: dict[str, float], tie_tolerance: float = _TIE_TOLERANCE) -> str:
    """Pick the dominant state with engaged-first and curiosity-last tie handling."""
    max_value = max(results.values())
    tied_states = [
        state for state, value in results.items() if max_value - float(value) <= tie_tolerance
    ]
    if "engaged" in tied_states:
        return "engaged"
    for state in ("overwhelmed", "distracted", "impatient", "curiosity", "engaged"):
        if state in tied_states:
            return state
    raise RuntimeError("Failed to resolve dominant state from fuzzy outputs.")


def _fuzzify_inputs(
    system: FuzzySystemDefinition,
    *,
    following_time: float,
    hhd: float,
    hrd: float,
    density: float,
    angle: float,
) -> dict[str, dict[str, float]]:
    crisp_inputs = {
        "following_time": float(following_time),
        "hhd": float(hhd),
        "hrd": float(hrd),
        "density": float(density),
        "angle": float(angle),
    }
    return {
        var_name: {
            term_name: float(
                np.interp(
                    crisp_inputs[var_name],
                    _INPUT_UNIVERSES[var_name],
                    system.input_curves[var_name][term_name],
                )
            )
            for term_name in term_specs.keys()
        }
        for var_name, term_specs in system.input_specs.items()
    }


def _evaluate_rule_strength(
    input_memberships: dict[str, dict[str, float]],
    antecedents: tuple[tuple[str, str], ...],
) -> float:
    strength = 1.0
    for var_name, term_name in antecedents:
        strength = min(strength, float(input_memberships[var_name][term_name]))
        if strength <= 0.0:
            return 0.0
    return strength


def _defuzzify_output(
    output_curves: dict[str, np.ndarray],
    term_strengths: dict[str, float],
    *,
    default_value: float,
) -> float:
    clipped_curves = []
    for term_name in _OUTPUT_TERMS:
        strength = float(term_strengths[term_name])
        if strength <= 0.0:
            continue
        clipped_curves.append(np.minimum(output_curves[term_name], strength))

    if not clipped_curves:
        return float(default_value)

    aggregated = np.maximum.reduce(clipped_curves)
    denominator = float(np.sum(aggregated))
    if denominator <= 1e-12:
        return float(default_value)
    numerator = float(np.sum(_OUTPUT_UNIVERSE * aggregated))
    return numerator / denominator


def _evaluate_system(
    system: FuzzySystemDefinition,
    *,
    following_time: float,
    hhd: float,
    hrd: float,
    density: float,
    angle: float,
) -> dict[str, float]:
    input_memberships = _fuzzify_inputs(
        system,
        following_time=following_time,
        hhd=hhd,
        hrd=hrd,
        density=density,
        angle=angle,
    )
    output_term_strengths = {
        output_name: {term_name: 0.0 for term_name in _OUTPUT_TERMS}
        for output_name in _OUTPUT_NAMES
    }

    for rule in system.rules:
        strength = _evaluate_rule_strength(input_memberships, rule.antecedents)
        if strength <= 0.0:
            continue
        current = output_term_strengths[rule.consequent_name][rule.consequent_term]
        if strength > current:
            output_term_strengths[rule.consequent_name][rule.consequent_term] = strength

    results = {
        output_name: _defuzzify_output(
            system.output_curves,
            output_term_strengths[output_name],
            default_value=_OUTPUT_DEFAULTS[output_name],
        )
        for output_name in _OUTPUT_NAMES
    }
    dominant = _select_dominant_state(results)
    results["dominant_state"] = dominant
    results["dominant_value"] = results[dominant]
    return results


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
    Run the fast fuzzy inference system for a single input vector.

    Returns a dict with keys:
    overwhelmed, distracted, impatient, engaged, curiosity, dominant_state, dominant_value.
    """
    normalized_context = _normalize_context(context)
    normalized_profile = _normalize_profile(profile)
    system = _SYSTEMS[(normalized_context, normalized_profile)]
    return _evaluate_system(
        system,
        following_time=following_time,
        hhd=hhd,
        hrd=hrd,
        density=density,
        angle=angle,
    )


def compute_batch(
    inputs: np.ndarray,
    *,
    context: str,
    profile: str = HumanProfile.NORMAL,
) -> list[dict]:
    """Run inference on multiple input vectors using the shared fast backend."""
    rows = np.asarray(inputs, dtype=np.float64)
    normalized_context = _normalize_context(context)
    normalized_profile = _normalize_profile(profile)
    system = _SYSTEMS[(normalized_context, normalized_profile)]
    return [
        _evaluate_system(
            system,
            following_time=float(row[0]),
            hhd=float(row[1]),
            hrd=float(row[2]),
            density=float(row[3]),
            angle=float(row[4]),
        )
        for row in rows
    ]


@lru_cache(maxsize=4)
def _get_reference_system(context: str, profile: str):
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl

    normalized_context = _normalize_context(context)
    normalized_profile = _normalize_profile(profile)
    input_specs = _build_input_specs(normalized_context, normalized_profile)

    ft = ctrl.Antecedent(np.linspace(0, 60, _RES), "following_time")
    hhd = ctrl.Antecedent(np.linspace(0, 4, _RES), "hhd")
    hrd = ctrl.Antecedent(np.linspace(0, 4, _RES), "hrd")
    density = ctrl.Antecedent(np.linspace(0, 10, _RES), "density")
    angle = ctrl.Antecedent(np.linspace(-180, 180, _RES), "angle")

    for term_name, spec in input_specs["following_time"].items():
        ft[term_name] = _mf(ft.universe, spec)
    for term_name, spec in input_specs["hhd"].items():
        hhd[term_name] = _mf(hhd.universe, spec)
    for term_name, spec in input_specs["hrd"].items():
        hrd[term_name] = _mf(hrd.universe, spec)
    for term_name, spec in input_specs["density"].items():
        density[term_name] = _mf(density.universe, spec)
    for term_name, spec in input_specs["angle"].items():
        angle[term_name] = _mf(angle.universe, spec)

    engaged = ctrl.Consequent(np.linspace(0, 1, _RES), "engaged", defuzzify_method="centroid")
    overwhelmed = ctrl.Consequent(np.linspace(0, 1, _RES), "overwhelmed", defuzzify_method="centroid")
    distracted = ctrl.Consequent(np.linspace(0, 1, _RES), "distracted", defuzzify_method="centroid")
    impatient = ctrl.Consequent(np.linspace(0, 1, _RES), "impatient", defuzzify_method="centroid")
    curiosity = ctrl.Consequent(np.linspace(0, 1, _RES), "curiosity", defuzzify_method="centroid")

    consequents = {
        "engaged": engaged,
        "overwhelmed": overwhelmed,
        "distracted": distracted,
        "impatient": impatient,
        "curiosity": curiosity,
    }
    for consequent in consequents.values():
        consequent["low"] = _SHARED_OUTPUT_CURVES["low"]
        consequent["medium"] = _SHARED_OUTPUT_CURVES["medium"]
        consequent["high"] = _SHARED_OUTPUT_CURVES["high"]

    antecedent_map = {
        "following_time": ft,
        "hhd": hhd,
        "hrd": hrd,
        "density": density,
        "angle": angle,
    }

    rules = []
    for rule in _SHARED_RULES:
        antecedent_expr = None
        for var_name, term_name in rule.antecedents:
            current = antecedent_map[var_name][term_name]
            antecedent_expr = current if antecedent_expr is None else antecedent_expr & current
        rules.append(
            ctrl.Rule(
                antecedent_expr,
                consequents[rule.consequent_name][rule.consequent_term],
            )
        )
    return ctrl.ControlSystem(rules)


def compute_reference(
    following_time: float,
    hhd: float,
    hrd: float,
    density: float,
    angle: float,
    *,
    context: str,
    profile: str = HumanProfile.NORMAL,
) -> dict:
    """Run the original scikit-fuzzy backend for regression tests."""
    from skfuzzy import control as ctrl

    normalized_context = _normalize_context(context)
    normalized_profile = _normalize_profile(profile)
    simulation = ctrl.ControlSystemSimulation(
        _get_reference_system(normalized_context, normalized_profile)
    )
    simulation.input["following_time"] = float(following_time)
    simulation.input["hhd"] = float(hhd)
    simulation.input["hrd"] = float(hrd)
    simulation.input["density"] = float(density)
    simulation.input["angle"] = float(angle)
    simulation.compute()

    results = {
        "overwhelmed": simulation.output.get("overwhelmed", _DEFAULT),
        "distracted": simulation.output.get("distracted", _DEFAULT),
        "impatient": simulation.output.get("impatient", _DEFAULT),
        "engaged": simulation.output.get("engaged", _DEFAULT),
        "curiosity": simulation.output.get("curiosity", _CURIOSITY_DEFAULT),
    }
    dominant = _select_dominant_state(results)
    results["dominant_state"] = dominant
    results["dominant_value"] = results[dominant]
    return results


def compute_reference_batch(
    inputs: np.ndarray,
    *,
    context: str,
    profile: str = HumanProfile.NORMAL,
) -> list[dict]:
    rows = np.asarray(inputs, dtype=np.float64)
    return [
        compute_reference(
            following_time=float(row[0]),
            hhd=float(row[1]),
            hrd=float(row[2]),
            density=float(row[3]),
            angle=float(row[4]),
            context=context,
            profile=profile,
        )
        for row in rows
    ]


if __name__ == "__main__":
    sample_input = (3.1, 0.56, 1.05, 5.0, 0.0)

    for sample_context in _CONTEXTS:
        result = compute(*sample_input, context=sample_context)
        print(
            f"{sample_context:<10} "
            f"dominant={result['dominant_state']:<12} "
            f"value={result['dominant_value']:.3f}"
        )
