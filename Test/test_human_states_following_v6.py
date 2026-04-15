import ast
import importlib.util
import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
FIS_PATH = REPO_ROOT / "Matlab fuzzy" / "human_states_following_v6.fis"
FUZZY_RULES_PATH = REPO_ROOT / "museum_env_package" / "Fuzzy_rules" / "human_states_following.py"
ENV_FUZZY_PATH = (
    REPO_ROOT / "museum_env_package" / "museum_env" / "fuzzy" / "human_states_following.py"
)

EXPECTED_INPUT_MFS = {
    ("hhd", "medium"): ("trapmf", (0.5, 0.6, 1.2, 1.5)),
    ("hhd", "far"): ("trapmf", (1.2, 1.5, 4.0, 4.0)),
    ("hrd", "medium"): ("trapmf", (1.0, 1.2, 2.6, 3.0)),
    ("hrd", "far"): ("trapmf", (2.6, 3.0, 5.0, 5.0)),
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
        if variable_name not in {"hhd", "hrd"}:
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
        if current_input not in {"hhd", "hrd"} or not line.startswith("MF"):
            continue
        match = re.match(r"MF\d+='([^']+)':'([^']+)',\[(.+)\]", line)
        if match is None:
            raise AssertionError(f"Could not parse MF line: {line}")
        label, shape, values_text = match.groups()
        values = tuple(float(token) for token in values_text.split())
        inputs[(current_input, label)] = (shape, values)
    return inputs


class TestHumanStatesFollowingV6(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fuzzy_rules_module = _load_module("fuzzy_rules_hsf_v6", FUZZY_RULES_PATH)
        cls.env_fuzzy_module = _load_module("env_fuzzy_hsf_v6", ENV_FUZZY_PATH)

    def test_rule_tables_match_v6_fis_exactly(self):
        fis_rules = _parse_fis_rules(FIS_PATH)
        fuzzy_rules_table = _load_rule_table_from_source(FUZZY_RULES_PATH, "_RULE_TABLE")
        env_rules_table = _load_rule_table_from_source(ENV_FUZZY_PATH, "RULE_TABLE")

        self.assertEqual(fuzzy_rules_table, fis_rules)
        self.assertEqual(env_rules_table, fis_rules)

    def test_updated_input_membership_functions_match_v6_fis(self):
        fis_input_mfs = _parse_fis_input_mfs(FIS_PATH)
        fuzzy_rules_input_mfs = _load_input_mfs_from_source(FUZZY_RULES_PATH)
        env_fuzzy_input_mfs = _load_input_mfs_from_source(ENV_FUZZY_PATH)

        for key, expected in EXPECTED_INPUT_MFS.items():
            self.assertEqual(fis_input_mfs[key], expected)
            self.assertEqual(fuzzy_rules_input_mfs[key], expected)
            self.assertEqual(env_fuzzy_input_mfs[key], expected)

    def test_distracted_outputs_follow_corrected_v6_patterns(self):
        module = self.env_fuzzy_module
        distracted_cases = [
            ((5.0, 2.0, 3.5, 0.5), 0.0, 0.4),
            ((30.0, 2.0, 2.0, 7.0), 0.3, 0.7),
            ((50.0, 2.0, 3.5, 7.0), 0.3, 0.72),
            ((50.0, 2.5, 3.5, 7.0), 0.6, 1.0),
        ]

        for inputs, lower, upper in distracted_cases:
            with self.subTest(inputs=inputs):
                result = module.compute(*inputs)
                self.assertGreaterEqual(result["distracted"], lower)
                self.assertLessEqual(result["distracted"], upper)

    def test_both_python_implementations_produce_matching_outputs(self):
        sample_inputs = [
            (5.0, 2.0, 3.5, 0.5),
            (30.0, 1.0, 2.0, 7.0),
            (30.0, 1.2, 2.6, 4.0),
            (30.0, 1.5, 3.0, 5.0),
            (50.0, 2.0, 3.5, 7.0),
            (50.0, 2.5, 3.5, 7.0),
        ]

        for inputs in sample_inputs:
            with self.subTest(inputs=inputs):
                first = self.fuzzy_rules_module.compute(*inputs)
                second = self.env_fuzzy_module.compute(*inputs)
                for key in ("overwhelmed", "distracted", "impatient", "engaged", "dominant_value"):
                    self.assertAlmostEqual(first[key], second[key], places=6)
                self.assertEqual(first["dominant_state"], second["dominant_state"])


if __name__ == "__main__":
    unittest.main()
