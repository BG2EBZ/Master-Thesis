import re
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "museum_env_package"
for path in (REPO_ROOT, PACKAGE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from museum_env.fuzzy import human_states
from museum_env.human import HumanProfile


FIS_DIR = PACKAGE_ROOT / "museum_env" / "fuzzy" / "matlab"
OUTPUT_MF_SPECS = {
    "low": ("trapmf", [0.0, 0.0, 0.2, 0.5]),
    "medium": ("trimf", [0.2, 0.5, 0.8]),
    "high": ("trapmf", [0.5, 0.8, 1.0, 1.0]),
}
SYSTEM_FIELDS = {
    "Type": "mamdani",
    "AndMethod": "min",
    "OrMethod": "max",
    "ImpMethod": "min",
    "AggMethod": "max",
    "DefuzzMethod": "centroid",
}


def _parse_scalar(value: str):
    text = value.strip()
    if text.startswith("'") and text.endswith("'"):
        return text[1:-1]
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    return float(text)


def _parse_range(value: str) -> list[float]:
    text = value.strip()
    if not (text.startswith("[") and text.endswith("]")):
        raise ValueError(f"Invalid range: {value!r}")
    return [float(part) for part in text[1:-1].split()]


def _parse_mf(line: str) -> dict:
    match = re.fullmatch(r"MF(\d+)='([^']+)':'([^']+)',\[(.*)\]", line.strip())
    if match is None:
        raise ValueError(f"Invalid MF line: {line!r}")
    points = [float(part) for part in match.group(4).split()]
    return {
        "index": int(match.group(1)),
        "name": match.group(2),
        "type": match.group(3),
        "points": points,
    }


def _parse_rule(line: str) -> dict:
    match = re.fullmatch(r"([0-9 ]+), ([0-9 ]+) \(([^)]+)\) : ([0-9]+)", line.strip())
    if match is None:
        raise ValueError(f"Invalid rule line: {line!r}")
    return {
        "antecedents": [int(part) for part in match.group(1).split()],
        "consequents": [int(part) for part in match.group(2).split()],
        "weight": float(match.group(3)),
        "connection": int(match.group(4)),
    }


def _parse_fis(path: Path) -> dict:
    system = {}
    inputs = []
    outputs = []
    rules = []
    current_section = None
    current_block = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            current_section = line[1:-1]
            if current_section.startswith("Input"):
                current_block = {"mfs": []}
                inputs.append(current_block)
            elif current_section.startswith("Output"):
                current_block = {"mfs": []}
                outputs.append(current_block)
            else:
                current_block = None
            continue
        if current_section == "System":
            key, value = line.split("=", 1)
            system[key] = _parse_scalar(value)
            continue
        if current_section == "Rules":
            rules.append(_parse_rule(line))
            continue
        if current_section and current_section.startswith(("Input", "Output")):
            if line.startswith("MF"):
                current_block["mfs"].append(_parse_mf(line))
            else:
                key, value = line.split("=", 1)
                if key == "Range":
                    current_block[key] = _parse_range(value)
                else:
                    current_block[key] = _parse_scalar(value)
            continue
        raise ValueError(f"Unexpected line {line!r} in section {current_section!r}")

    return {"system": system, "inputs": inputs, "outputs": outputs, "rules": rules}


def _expected_rule_rows(profile: str) -> list[dict]:
    system = human_states._SYSTEMS[("following", profile)]
    input_specs = system.input_specs
    input_term_indices = {
        input_name: {term_name: idx for idx, term_name in enumerate(term_specs.keys(), start=1)}
        for input_name, term_specs in input_specs.items()
    }
    output_term_indices = {
        output_name: {term_name: idx for idx, term_name in enumerate(human_states._OUTPUT_TERMS, start=1)}
        for output_name in human_states._OUTPUT_NAMES
    }

    expected = []
    for rule in system.rules:
        antecedents = []
        for input_name in input_specs.keys():
            term_name = next(
                (term for var_name, term in rule.antecedents if var_name == input_name),
                None,
            )
            antecedents.append(input_term_indices[input_name][term_name] if term_name else 0)
        consequents = []
        for output_name in human_states._OUTPUT_NAMES:
            if output_name == rule.consequent_name:
                consequents.append(output_term_indices[output_name][rule.consequent_term])
            else:
                consequents.append(0)
        expected.append(
            {
                "antecedents": antecedents,
                "consequents": consequents,
                "weight": 1.0,
                "connection": 1,
            }
        )
    return expected


class HumanStatesFisExportTest(unittest.TestCase):
    def test_fis_files_match_human_states_definitions(self):
        profiles = {
            HumanProfile.NORMAL: FIS_DIR / "normal.fis",
            HumanProfile.NEURODIVERGENT: FIS_DIR / "neurodivergent.fis",
        }

        for profile, path in profiles.items():
            with self.subTest(profile=profile):
                self.assertTrue(path.exists(), f"Missing FIS file: {path}")
                parsed = _parse_fis(path)
                expected_system = human_states._SYSTEMS[("following", profile)]

                self.assertEqual(parsed["system"]["Name"], profile)
                self.assertEqual(parsed["system"]["NumInputs"], len(expected_system.input_specs))
                self.assertEqual(parsed["system"]["NumOutputs"], len(human_states._OUTPUT_NAMES))
                self.assertEqual(parsed["system"]["NumRules"], len(expected_system.rules))
                for field, expected in SYSTEM_FIELDS.items():
                    self.assertEqual(parsed["system"][field], expected)

                self.assertEqual(len(parsed["inputs"]), len(expected_system.input_specs))
                for input_block, (input_name, term_specs) in zip(
                    parsed["inputs"],
                    expected_system.input_specs.items(),
                ):
                    with self.subTest(profile=profile, input_name=input_name):
                        self.assertEqual(input_block["Name"], input_name)
                        self.assertEqual(
                            input_block["Range"],
                            [
                                float(human_states._INPUT_UNIVERSES[input_name][0]),
                                float(human_states._INPUT_UNIVERSES[input_name][-1]),
                            ],
                        )
                        self.assertEqual(input_block["NumMFs"], len(term_specs))
                        self.assertEqual(len(input_block["mfs"]), len(term_specs))
                        for mf_block, (term_name, spec) in zip(input_block["mfs"], term_specs.items()):
                            self.assertEqual(mf_block["name"], term_name)
                            self.assertEqual(
                                mf_block["type"],
                                "trapmf" if spec.shape == "trap" else "trimf",
                            )
                            self.assertEqual(mf_block["points"], [float(point) for point in spec.points])

                self.assertEqual(len(parsed["outputs"]), len(human_states._OUTPUT_NAMES))
                for output_block, output_name in zip(parsed["outputs"], human_states._OUTPUT_NAMES):
                    with self.subTest(profile=profile, output_name=output_name):
                        self.assertEqual(output_block["Name"], output_name)
                        self.assertEqual(output_block["Range"], [0.0, 1.0])
                        self.assertEqual(output_block["NumMFs"], len(human_states._OUTPUT_TERMS))
                        self.assertEqual(len(output_block["mfs"]), len(human_states._OUTPUT_TERMS))
                        for mf_block, term_name in zip(output_block["mfs"], human_states._OUTPUT_TERMS):
                            expected_type, expected_points = OUTPUT_MF_SPECS[term_name]
                            self.assertEqual(mf_block["name"], term_name)
                            self.assertEqual(mf_block["type"], expected_type)
                            self.assertEqual(mf_block["points"], expected_points)

                self.assertEqual(parsed["rules"], _expected_rule_rows(profile))


if __name__ == "__main__":
    unittest.main()
