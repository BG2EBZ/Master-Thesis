import ast
import importlib.util
import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
FIS_PATH = REPO_ROOT / "Matlab fuzzy" / "human_states_following_v8.fis"
ENV_FUZZY_PATH = REPO_ROOT / "museum_env_package" / "museum_env" / "fuzzy" / "human_states_following.py"

EXPECTED_INPUT_MFS = {
    ("hhd", "medium"): ("trapmf", (0.5, 0.6, 0.9, 1.0)),
    ("hhd", "far"): ("trapmf", (0.9, 1.0, 4.0, 4.0)),
    ("hrd", "medium"): ("trapmf", (1.0, 1.2, 2.0, 2.2)),
    ("hrd", "far"): ("trapmf", (2.0, 2.2, 5.0, 5.0)),
    ("density", "low"): ("trapmf", (0.0, 0.0, 1.0, 1.0)),
    ("density", "medium"): ("trapmf", (2.0, 2.0, 4.0, 4.0)),
    ("density", "crowded"): ("trapmf", (5.0, 5.0, 10.0, 10.0)),
}


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_rule_table_from_source(path: Path, variable_name: str):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == variable_name:
                return ast.literal_eval(node.value)
    raise AssertionError(f"Could not find {variable_name} in {path}")


def _load_input_mfs_from_source(path: Path):
    tree = ast.parse(path.read_text(), filename=str(path))
    params = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Subscript) or not isinstance(target.value, ast.Name):
            continue
        variable_name = target.value.id
        if variable_name not in {"following_time", "hhd", "hrd", "density"}:
            continue
        label = target.slice.value if isinstance(target.slice, ast.Constant) else None
        if not isinstance(label, str):
            continue
        value = node.value
        if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Attribute):
            continue
        if value.func.attr not in {"trapmf", "trimf"}:
            continue
        shape = value.func.attr
        coordinates = tuple(float(ast.literal_eval(arg)) for arg in value.args[1].elts)
        params[(variable_name, label)] = (shape, coordinates)
    return params


def _parse_fis_rules(path: Path):
    rules = []
    in_rules = False
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if line == "[Rules]":
            in_rules = True
            continue
        if not in_rules or not line:
            continue
        antecedent_text, consequent_text = line.split(",", maxsplit=1)
        antecedent = tuple(int(token) for token in antecedent_text.split())
        consequent_match = re.search(r"([0-3] [0-3] [0-3] [0-3])", consequent_text)
        if consequent_match is None:
            raise AssertionError(f"Could not parse consequent from FIS rule: {line}")
        consequent = tuple(int(token) for token in consequent_match.group(1).split())
        rules.append(antecedent + consequent)
    return rules


def _parse_fis_input_mfs(path: Path):
    current_input = None
    inputs = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if line.startswith("[Input"):
            current_input = None
            continue
        if line.startswith("Name='") and current_input is None:
            current_input = line.split("'")[1]
            continue
        if current_input not in {"following_time", "hhd", "hrd", "density"} or not line.startswith("MF"):
            continue
        match = re.match(r"MF\d+='([^']+)':'([^']+)',\[(.+)\]", line)
        if match is None:
            raise AssertionError(f"Could not parse MF line: {line}")
        label, shape, values_text = match.groups()
        values = tuple(float(token) for token in values_text.split())
        inputs[(current_input, label)] = (shape, values)
    return inputs


class TestHumanStatesFollowingV8(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env_fuzzy_module = _load_module("env_fuzzy_hsf_v8", ENV_FUZZY_PATH)

    def test_rule_table_matches_v8_fis_exactly(self):
        fis_rules = _parse_fis_rules(FIS_PATH)
        env_rules_table = _load_rule_table_from_source(ENV_FUZZY_PATH, "_RULE_TABLE")
        self.assertEqual(env_rules_table, fis_rules)

    def test_updated_input_membership_functions_match_v8_fis(self):
        fis_input_mfs = _parse_fis_input_mfs(FIS_PATH)
        env_fuzzy_input_mfs = _load_input_mfs_from_source(ENV_FUZZY_PATH)

        for key, expected in EXPECTED_INPUT_MFS.items():
            self.assertEqual(fis_input_mfs[key], expected)
            self.assertEqual(env_fuzzy_input_mfs[key], expected)

    def test_representative_v8_behavior_cases(self):
        module = self.env_fuzzy_module
        cases = [
            ((30.0, 0.25, 0.5, 7.0), "overwhelmed", "overwhelmed", 0.55),
            ((55.0, 2.0, 3.5, 0.5), "distracted", "distracted", 0.55),
            ((55.0, 0.25, 0.5, 3.0), "impatient", "impatient", 0.55),
            ((5.0, 2.0, 3.5, 0.5), "engaged", "engaged", 0.55),
            ((30.0, 0.75, 0.5, 7.0), "engaged", "engaged", 0.35),
            ((55.0, 0.25, 0.5, 0.5), "engaged", "engaged", 0.45),
        ]

        for inputs, dominant_state, key, lower_bound in cases:
            with self.subTest(inputs=inputs):
                result = module.compute(*inputs)
                self.assertEqual(result["dominant_state"], dominant_state)
                self.assertGreaterEqual(result[key], lower_bound)

    def test_no_rule_outputs_fall_back_to_point_two(self):
        result = self.env_fuzzy_module.compute(5.0, 2.0, 3.5, 0.5)
        self.assertAlmostEqual(result["overwhelmed"], 0.2, places=6)
        self.assertAlmostEqual(result["distracted"], 0.2, places=6)
        self.assertAlmostEqual(result["impatient"], 0.2, places=6)
        self.assertGreater(result["engaged"], 0.6)

    def test_tie_within_tolerance_prefers_engaged(self):
        dominant = self.env_fuzzy_module._select_dominant_state(
            {
                "overwhelmed": 0.705,
                "distracted": 0.1,
                "impatient": 0.2,
                "engaged": 0.7,
            }
        )
        self.assertEqual(dominant, "engaged")


if __name__ == "__main__":
    unittest.main()
