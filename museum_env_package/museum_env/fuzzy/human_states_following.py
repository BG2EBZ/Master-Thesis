"""
human_states_following_v6.py
Scikit-fuzzy implementation of human_states_following_v6.fis

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
    result = compute(following_time=30, hhd=2, hrd=2, density=5)
    print(result)
    # {'overwhelmed': 0.5, 'distracted': 0.847, 'impatient': 0.5, 'engaged': 0.153}
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


# ─────────────────────────────────────────────
# Universe of discourse
# ─────────────────────────────────────────────
_RES = 1000   # resolution — increase for higher precision, decrease for speed

following_time = ctrl.Antecedent(np.linspace(0, 60, _RES),  'following_time')
hhd            = ctrl.Antecedent(np.linspace(0, 4,  _RES),  'hhd')
hrd            = ctrl.Antecedent(np.linspace(0, 4,  _RES),  'hrd')
density        = ctrl.Antecedent(np.linspace(0, 10, _RES),  'density')

overwhelmed = ctrl.Consequent(np.linspace(0, 1, _RES), 'overwhelmed', defuzzify_method='centroid')
distracted  = ctrl.Consequent(np.linspace(0, 1, _RES), 'distracted',  defuzzify_method='centroid')
impatient   = ctrl.Consequent(np.linspace(0, 1, _RES), 'impatient',   defuzzify_method='centroid')
engaged     = ctrl.Consequent(np.linspace(0, 1, _RES), 'engaged',     defuzzify_method='centroid')


# ─────────────────────────────────────────────
# Input membership functions
# ─────────────────────────────────────────────
following_time['short']  = fuzz.trapmf(following_time.universe, [0,  0,  10, 20])
following_time['medium'] = fuzz.trapmf(following_time.universe, [10, 20, 40, 50])
following_time['long']   = fuzz.trapmf(following_time.universe, [40, 50, 60, 60])

hhd['close']  = fuzz.trapmf(hhd.universe, [0,   0,   0.5, 0.6])
hhd['medium'] = fuzz.trapmf(hhd.universe, [0.5, 0.6, 1.2, 1.5])
hhd['far']    = fuzz.trapmf(hhd.universe, [1.2, 1.5, 4.0, 4.0])

hrd['close']  = fuzz.trapmf(hrd.universe, [0,   0,   1.0, 1.2])
hrd['medium'] = fuzz.trapmf(hrd.universe, [1.0, 1.2, 2.6, 3.0])
hrd['far']    = fuzz.trapmf(hrd.universe, [2.6, 3.0, 5.0, 5.0])

density['low']     = fuzz.trapmf(density.universe, [0, 0, 1,  1 ])
density['medium']  = fuzz.trapmf(density.universe, [2, 2, 4,  4 ])
density['crowded'] = fuzz.trapmf(density.universe, [5, 5, 10, 10])


# ─────────────────────────────────────────────
# Output membership functions (identical for all 4 outputs)
# ─────────────────────────────────────────────
for _var in [overwhelmed, distracted, impatient, engaged]:
    _var['low']    = fuzz.trapmf(_var.universe, [0,   0,   0.2, 0.4])
    _var['medium'] = fuzz.trimf( _var.universe, [0.3, 0.5, 0.7])
    _var['high']   = fuzz.trapmf(_var.universe, [0.6, 0.8, 1.0, 1.0])


# ─────────────────────────────────────────────
# Rules  (48 total, matching human_states_following_v6.fis exactly)
# Encoding: 0=don't care, 1/2/3 = MF index
# Input order:  [following_time, hhd, hrd, density]
# Output order: [overwhelmed, distracted, impatient, engaged]
# ─────────────────────────────────────────────

# Shorthand aliases
ft  = following_time
_mf = {
    'ft':  {1: ft['short'],  2: ft['medium'],  3: ft['long']},
    'hhd': {1: hhd['close'], 2: hhd['medium'], 3: hhd['far']},
    'hrd': {1: hrd['close'], 2: hrd['medium'], 3: hrd['far']},
    'den': {1: density['low'], 2: density['medium'], 3: density['crowded']},
    'ovw': {1: overwhelmed['low'], 2: overwhelmed['medium'], 3: overwhelmed['high']},
    'dis': {1: distracted['low'],  2: distracted['medium'],  3: distracted['high']},
    'imp': {1: impatient['low'],   2: impatient['medium'],   3: impatient['high']},
    'eng': {1: engaged['low'],     2: engaged['medium'],     3: engaged['high']},
}

def _make_rules(ft_i, hhd_i, hrd_i, den_i, ovw_i, dis_i, imp_i, eng_i):
    """
    Build skfuzzy Rules from FIS integer encoding (0 = don't care).
    skfuzzy does not support multi-consequent rules, so each output
    consequent becomes a separate Rule sharing the same antecedent.
    Returns a list of Rule objects (1 per non-zero output).
    """
    antecedents = []
    # ignore zeroes and build list of antecedent MFs
    if ft_i:  antecedents.append(_mf['ft'][ft_i])
    if hhd_i: antecedents.append(_mf['hhd'][hhd_i])
    if hrd_i: antecedents.append(_mf['hrd'][hrd_i])
    if den_i: antecedents.append(_mf['den'][den_i])

    ant = antecedents[0]
    # Fuzzy AND (min) of all antecedents
    for a in antecedents[1:]:
        ant = ant & a

    rules_out = []
    if ovw_i: rules_out.append(ctrl.Rule(ant, _mf['ovw'][ovw_i]))
    if dis_i: rules_out.append(ctrl.Rule(ant, _mf['dis'][dis_i]))
    if imp_i: rules_out.append(ctrl.Rule(ant, _mf['imp'][imp_i]))
    if eng_i: rules_out.append(ctrl.Rule(ant, _mf['eng'][eng_i]))
    return rules_out


# FIS rule table: (ft, hhd, hrd, den,  ovw, dis, imp, eng)
_RULE_TABLE = [
    # ── overwhelmed rules ─────────────────────────────────────────────
    (0, 1, 0, 3,  3, 0, 0, 0),  # R01  hhd=close + crowded → ovw=high
    (3, 0, 0, 3,  3, 0, 0, 0),  # R02  long + crowded → ovw=high
    (0, 2, 0, 3,  2, 0, 0, 0),  # R03  hhd=medium + crowded → ovw=medium
    (3, 1, 0, 2,  2, 0, 0, 0),  # R04  long + hhd=close + medium → ovw=medium
    (0, 2, 0, 2,  1, 0, 0, 0),  # R05  hhd=medium + medium → ovw=low
    (0, 3, 0, 1,  1, 0, 0, 0),  # R06  hhd=far + low → ovw=low
    (1, 2, 0, 1,  1, 0, 0, 0),  # R07  short + hhd=medium + low → ovw=low
    (3, 1, 0, 1,  2, 0, 0, 0),  # R08  long + hhd=close + low → ovw=medium
    # ── distracted rules ──────────────────────────────────────────────
    (3, 3, 3, 3,  0, 3, 0, 0),  # R09  long + hhd=far + hrd=far + crowded → dis=high
    (2, 0, 3, 3,  0, 2, 0, 0),  # R10  medium + hrd=far + crowded → dis=medium
    (3, 3, 3, 0,  0, 3, 0, 0),  # R11  long + hhd=far + hrd=far → dis=high
    (3, 0, 3, 3,  0, 2, 0, 0),  # R12  long + hrd=far + crowded → dis=medium
    (0, 0, 2, 3,  0, 2, 0, 0),  # R13  hrd=medium + crowded → dis=medium
    (2, 0, 2, 1,  0, 1, 0, 0),  # R14  medium + hrd=medium + low → dis=low
    (2, 0, 1, 0,  0, 1, 0, 0),  # R15  medium + hrd=close → dis=low
    (3, 0, 1, 1,  0, 1, 0, 0),  # R16  long + hrd=close + low → dis=low
    (1, 0, 3, 0,  0, 1, 0, 0),  # R17  short + hrd=far → dis=low
    (2, 0, 3, 1,  0, 1, 0, 0),  # R18  medium + hrd=far + low → dis=low
    # ── impatient rules ───────────────────────────────────────────────
    (3, 0, 3, 0,  0, 0, 3, 0),  # R19  long + hrd=far → imp=high
    (3, 0, 0, 3,  0, 0, 3, 0),  # R20  long + crowded → imp=high
    (3, 1, 0, 0,  0, 0, 3, 0),  # R21  long + hhd=close → imp=high
    (2, 0, 2, 3,  0, 0, 2, 0),  # R22  medium + hrd=medium + crowded → imp=medium
    (3, 0, 2, 1,  0, 0, 2, 0),  # R23  long + hrd=medium + low → imp=medium
    (1, 0, 0, 0,  0, 0, 1, 0),  # R24  short → imp=low
    (2, 0, 1, 1,  0, 0, 1, 0),  # R25  medium + hrd=close + low → imp=low
    (2, 0, 2, 2,  0, 0, 1, 0),  # R26  medium + hrd=medium + medium → imp=low
    # ── engaged=high rules ────────────────────────────────────────────
    (1, 0, 0, 1,  0, 0, 0, 3),  # R27  short + low → eng=high
    (1, 0, 0, 2,  0, 0, 0, 3),  # R28  short + medium → eng=high
    (2, 0, 1, 1,  0, 0, 0, 3),  # R29  medium + hrd=close + low → eng=high
    (2, 0, 1, 2,  0, 0, 0, 3),  # R30  medium + hrd=close + medium → eng=high
    (2, 0, 2, 1,  0, 0, 0, 3),  # R31  medium + hrd=medium + low → eng=high
    (3, 3, 1, 1,  0, 0, 0, 3),  # R32  long + hhd=far + hrd=close + low → eng=high
    (1, 3, 2, 1,  0, 0, 0, 3),  # R33  short + hhd=far + hrd=medium + low → eng=high
    # ── engaged=medium rules ──────────────────────────────────────────
    (2, 0, 2, 2,  0, 0, 0, 2),  # R34  medium + hrd=medium + medium → eng=medium
    (3, 3, 1, 2,  0, 0, 0, 2),  # R35  long + hhd=far + hrd=close + medium → eng=medium
    (3, 0, 2, 1,  0, 0, 0, 2),  # R36  long + hrd=medium + low → eng=medium
    (3, 0, 1, 1,  0, 0, 0, 2),  # R37  long + hrd=close + low → eng=medium
    (1, 3, 3, 1,  0, 0, 0, 2),  # R38  short + hhd=far + hrd=far + low → eng=medium
    (2, 3, 1, 2,  0, 0, 0, 2),  # R39  medium + hhd=far + hrd=close + medium → eng=medium
    (1, 3, 2, 2,  0, 0, 0, 2),  # R40  short + hhd=far + hrd=medium + medium → eng=medium
    (3, 0, 2, 2,  0, 0, 0, 2),  # R41  long + hrd=medium + medium → eng=medium
    (3, 2, 0, 2,  0, 0, 0, 2),  # R42  long + hhd=medium + medium → eng=medium
    # ── engaged=low rules (互斥/兜底压制) ─────────────────────────────
    (0, 0, 0, 3,  0, 0, 0, 1),  # R43  crowded → eng=low
    (3, 0, 3, 0,  0, 0, 0, 1),  # R44  long + hrd=far → eng=low
    (0, 1, 0, 3,  0, 0, 0, 1),  # R45  hhd=close + crowded → eng=low
    (3, 0, 0, 3,  0, 0, 0, 1),  # R46  long + crowded → eng=low
    # ── engaged default fallback ──────────────────────────────────────
    (0, 0, 0, 1,  0, 0, 0, 3),  # R47  density=low → eng=high  (fallback)
    (0, 0, 0, 2,  0, 0, 0, 3),  # R48  density=medium → eng=high (fallback)
]

rules = [rule for row in _RULE_TABLE for rule in _make_rules(*row)]


# ─────────────────────────────────────────────
# Control system
# ─────────────────────────────────────────────
_system     = ctrl.ControlSystem(rules)
_simulation = ctrl.ControlSystemSimulation(_system)


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────
def compute(following_time: float,
            hhd: float,
            hrd: float,
            density: float) -> dict:
    """
    Run the fuzzy inference system for a single input vector.

    Parameters
    ----------
    following_time : float  [0, 60]
    hhd            : float  [0, 4]
    hrd            : float  [0, 4]
    density        : float  [0, 10]

    Returns
    -------
    dict with keys: overwhelmed, distracted, impatient, engaged
                    and dominant_state (str), dominant_value (float)

    Note: if no rule fires for an output variable, its value defaults
    to 0.2 (centroid of the 'low' MF).
    """
    _simulation.input['following_time'] = float(following_time)
    _simulation.input['hhd']            = float(hhd)
    _simulation.input['hrd']            = float(hrd)
    _simulation.input['density']        = float(density)
    _simulation.compute()

    # Use .get() with a default of 0.2 (centroid of 'low' MF)
    # in case no rule fires for a particular output variable
    # 0.2 can filter out sense noise
    _DEFAULT = 0.2
    results = {
        'overwhelmed': _simulation.output.get('overwhelmed', _DEFAULT),
        'distracted':  _simulation.output.get('distracted',  _DEFAULT),
        'impatient':   _simulation.output.get('impatient',   _DEFAULT),
        'engaged':     _simulation.output.get('engaged',     _DEFAULT),
    }

    # Choose maximum output as dominant state
    dominant = max(results, key=results.get)
    results['dominant_state'] = dominant
    results['dominant_value'] = results[dominant]
    return results


def compute_batch(inputs: np.ndarray) -> list[dict]:
    """
    Run inference on multiple input vectors.

    Parameters
    ----------
    inputs : np.ndarray, shape (N, 4)
             columns: [following_time, hhd, hrd, density]

    Returns
    -------
    list of N dicts (same format as compute())
    """
    return [compute(*row) for row in inputs]


# ─────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────
if __name__ == '__main__':
    import pandas as pd

    test_cases = [
        # (ft,   hhd,  hrd,  density,  expected_dominant)
        (  5,  0.25,  0.5,    0.5,   'engaged'),      # short+close+close+low
        ( 30,  0.25,  0.5,    7.0,   'overwhelmed'),  # medium+close+close+crowded
        ( 30,  2.0,   3.5,    0.5,   'distracted'),   # medium+far+far+low
        ( 55,  0.25,  0.5,    0.5,   'impatient'),    # long+close+close+low
        ( 30,  2.0,   1.6,    0.5,   'engaged'),      # medium+far+medium+low
    ]

    print(f"{'ft':>5} {'hhd':>5} {'hrd':>5} {'den':>6}  "
          f"{'ovw':>6} {'dis':>6} {'imp':>6} {'eng':>6}  "
          f"{'dominant':<12} {'expected':<12} {'ok'}")
    print("─" * 85)

    for ft_v, hhd_v, hrd_v, den_v, expected in test_cases:
        r = compute(ft_v, hhd_v, hrd_v, den_v)
        ok = '✅' if r['dominant_state'] == expected else '⚠️ '
        print(f"{ft_v:>5} {hhd_v:>5} {hrd_v:>5} {den_v:>6}  "
              f"{r['overwhelmed']:>6.3f} {r['distracted']:>6.3f} "
              f"{r['impatient']:>6.3f} {r['engaged']:>6.3f}  "
              f"{r['dominant_state']:<12} {expected:<12} {ok}")

    # 81-combination sweep
    print("\n=== 81-combination sweep ===")
    ft_vals  = [5,    30,   55  ]
    hhd_vals = [0.25, 0.75,  2.0]
    hrd_vals = [0.5,  1.6,   3.5]
    den_vals = [0.5,  3.0,   7.0]
    ft_labels  = ['short',  'medium', 'long'   ]
    hhd_labels = ['close',  'medium', 'far'    ]
    hrd_labels = ['close',  'medium', 'far'    ]
    den_labels = ['low',    'medium', 'crowded']

    rows = []
    for i, fv in enumerate(ft_vals):
        for j, hv in enumerate(hhd_vals):
            for k, rv in enumerate(hrd_vals):
                for l, dv in enumerate(den_vals):
                    r = compute(fv, hv, rv, dv)
                    rows.append({
                        'following_time': ft_labels[i],
                        'hhd':           hhd_labels[j],
                        'hrd':           hrd_labels[k],
                        'density':       den_labels[l],
                        **{key: round(r[key], 4)
                           for key in ['overwhelmed','distracted','impatient','engaged']},
                        'dominant_state': r['dominant_state'],
                        'dominant_value': round(r['dominant_value'], 4),
                    })

    df = pd.DataFrame(rows)
    print(df['dominant_state'].value_counts().to_string())
    print(f"\nTotal: {len(df)} combinations")
    df.to_csv('fis_81_combinations_python.csv', index=False)
    print("Saved → fis_81_combinations_python.csv")
