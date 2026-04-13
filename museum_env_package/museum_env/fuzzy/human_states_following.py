import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

_RESOLUTION = 1000
_DEFAULT_OUTPUT_VALUE = 0.2
OUTPUT_STATES = ("overwhelmed", "distracted", "impatient", "engaged")


def _build_control_system():
    following_time = ctrl.Antecedent(np.linspace(0, 60, _RESOLUTION), "following_time")
    hhd = ctrl.Antecedent(np.linspace(0, 4, _RESOLUTION), "hhd")
    hrd = ctrl.Antecedent(np.linspace(0, 4, _RESOLUTION), "hrd")
    density = ctrl.Antecedent(np.linspace(0, 10, _RESOLUTION), "density")

    overwhelmed = ctrl.Consequent(
        np.linspace(0, 1, _RESOLUTION), "overwhelmed", defuzzify_method="centroid"
    )
    distracted = ctrl.Consequent(
        np.linspace(0, 1, _RESOLUTION), "distracted", defuzzify_method="centroid"
    )
    impatient = ctrl.Consequent(
        np.linspace(0, 1, _RESOLUTION), "impatient", defuzzify_method="centroid"
    )
    engaged = ctrl.Consequent(
        np.linspace(0, 1, _RESOLUTION), "engaged", defuzzify_method="centroid"
    )

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

    for output_var in (overwhelmed, distracted, impatient, engaged):
        output_var["low"] = fuzz.trapmf(output_var.universe, [0, 0, 0.2, 0.4])
        output_var["medium"] = fuzz.trimf(output_var.universe, [0.3, 0.5, 0.7])
        output_var["high"] = fuzz.trapmf(output_var.universe, [0.6, 0.8, 1.0, 1.0])

    mf = {
        "ft": {1: following_time["short"], 2: following_time["medium"], 3: following_time["long"]},
        "hhd": {1: hhd["close"], 2: hhd["medium"], 3: hhd["far"]},
        "hrd": {1: hrd["close"], 2: hrd["medium"], 3: hrd["far"]},
        "den": {1: density["low"], 2: density["medium"], 3: density["crowded"]},
        "ovw": {1: overwhelmed["low"], 2: overwhelmed["medium"], 3: overwhelmed["high"]},
        "dis": {1: distracted["low"], 2: distracted["medium"], 3: distracted["high"]},
        "imp": {1: impatient["low"], 2: impatient["medium"], 3: impatient["high"]},
        "eng": {1: engaged["low"], 2: engaged["medium"], 3: engaged["high"]},
    }

    def make_rules(ft_i, hhd_i, hrd_i, den_i, ovw_i, dis_i, imp_i, eng_i):
        antecedents = []
        if ft_i:
            antecedents.append(mf["ft"][ft_i])
        if hhd_i:
            antecedents.append(mf["hhd"][hhd_i])
        if hrd_i:
            antecedents.append(mf["hrd"][hrd_i])
        if den_i:
            antecedents.append(mf["den"][den_i])

        antecedent = antecedents[0]
        for current in antecedents[1:]:
            antecedent = antecedent & current

        rules = []
        if ovw_i:
            rules.append(ctrl.Rule(antecedent, mf["ovw"][ovw_i]))
        if dis_i:
            rules.append(ctrl.Rule(antecedent, mf["dis"][dis_i]))
        if imp_i:
            rules.append(ctrl.Rule(antecedent, mf["imp"][imp_i]))
        if eng_i:
            rules.append(ctrl.Rule(antecedent, mf["eng"][eng_i]))
        return rules

    rule_table = [
        (0, 1, 0, 3, 3, 0, 0, 0),
        (3, 0, 0, 3, 3, 0, 0, 0),
        (0, 2, 0, 3, 2, 0, 0, 0),
        (3, 1, 0, 2, 2, 0, 0, 0),
        (0, 2, 0, 2, 1, 0, 0, 0),
        (0, 3, 0, 1, 1, 0, 0, 0),
        (1, 2, 0, 1, 1, 0, 0, 0),
        (3, 1, 0, 1, 2, 0, 0, 0),
        (1, 0, 3, 0, 0, 3, 0, 0),
        (2, 0, 3, 0, 0, 2, 0, 0),
        (3, 0, 3, 0, 0, 3, 0, 0),
        (0, 0, 2, 3, 0, 3, 0, 0),
        (0, 0, 1, 3, 0, 2, 0, 0),
        (2, 0, 2, 1, 0, 1, 0, 0),
        (2, 0, 1, 0, 0, 1, 0, 0),
        (3, 0, 1, 1, 0, 1, 0, 0),
        (3, 0, 3, 0, 0, 0, 3, 0),
        (3, 0, 0, 3, 0, 0, 3, 0),
        (3, 1, 0, 0, 0, 0, 3, 0),
        (2, 0, 2, 3, 0, 0, 2, 0),
        (3, 0, 2, 1, 0, 0, 2, 0),
        (1, 0, 0, 0, 0, 0, 1, 0),
        (2, 0, 1, 1, 0, 0, 1, 0),
        (2, 0, 2, 2, 0, 0, 1, 0),
        (1, 0, 0, 1, 0, 0, 0, 3),
        (1, 0, 0, 2, 0, 0, 0, 3),
        (2, 0, 1, 1, 0, 0, 0, 3),
        (2, 0, 1, 2, 0, 0, 0, 3),
        (2, 0, 2, 1, 0, 0, 0, 3),
        (3, 3, 1, 1, 0, 0, 0, 3),
        (1, 3, 2, 1, 0, 0, 0, 3),
        (2, 0, 2, 2, 0, 0, 0, 2),
        (3, 3, 1, 2, 0, 0, 0, 2),
        (3, 0, 2, 1, 0, 0, 0, 2),
        (3, 0, 1, 1, 0, 0, 0, 2),
        (1, 3, 3, 1, 0, 0, 0, 2),
        (2, 3, 1, 2, 0, 0, 0, 2),
        (1, 3, 2, 2, 0, 0, 0, 2),
        (3, 0, 2, 2, 0, 0, 0, 2),
        (3, 2, 0, 2, 0, 0, 0, 2),
        (0, 0, 0, 3, 0, 0, 0, 1),
        (3, 0, 3, 0, 0, 0, 0, 1),
        (0, 1, 0, 3, 0, 0, 0, 1),
        (3, 0, 0, 3, 0, 0, 0, 1),
        (2, 0, 3, 0, 0, 0, 0, 1),
        (3, 1, 0, 2, 0, 0, 0, 1),
        (0, 0, 0, 1, 0, 0, 0, 3),
        (0, 0, 0, 2, 0, 0, 0, 3),
    ]

    rules = [rule for row in rule_table for rule in make_rules(*row)]
    return ctrl.ControlSystem(rules)


_CONTROL_SYSTEM = _build_control_system()


class FollowingFuzzyEngine:
    def __init__(self):
        self._system = _CONTROL_SYSTEM
        self._simulation = ctrl.ControlSystemSimulation(self._system)

    @staticmethod
    def clip_inputs(following_time: float, hhd: float, hrd: float, density: float):
        return {
            "following_time": float(np.clip(following_time, 0.0, 60.0)),
            "hhd": float(np.clip(hhd, 0.0, 4.0)),
            "hrd": float(np.clip(hrd, 0.0, 4.0)),
            "density": float(np.clip(density, 0.0, 10.0)),
        }

    def compute(self, following_time: float, hhd: float, hrd: float, density: float) -> dict:
        clipped = self.clip_inputs(
            following_time=following_time,
            hhd=hhd,
            hrd=hrd,
            density=density,
        )
        reset = getattr(self._simulation, "reset", None)
        if callable(reset):
            reset()

        self._simulation.input["following_time"] = clipped["following_time"]
        self._simulation.input["hhd"] = clipped["hhd"]
        self._simulation.input["hrd"] = clipped["hrd"]
        self._simulation.input["density"] = clipped["density"]
        self._simulation.compute()

        results = {
            state: float(self._simulation.output.get(state, _DEFAULT_OUTPUT_VALUE))
            for state in OUTPUT_STATES
        }
        dominant_state = max(OUTPUT_STATES, key=results.get)
        results["dominant_state"] = dominant_state
        results["dominant_value"] = float(results[dominant_state])
        return results

    def compute_batch(self, inputs: np.ndarray) -> list[dict]:
        return [self.compute(*row) for row in np.asarray(inputs, dtype=np.float32)]


_DEFAULT_ENGINE = FollowingFuzzyEngine()


def compute(following_time: float, hhd: float, hrd: float, density: float) -> dict:
    return _DEFAULT_ENGINE.compute(
        following_time=following_time,
        hhd=hhd,
        hrd=hrd,
        density=density,
    )


def compute_batch(inputs: np.ndarray) -> list[dict]:
    return _DEFAULT_ENGINE.compute_batch(inputs=inputs)
