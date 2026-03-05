import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = ROOT / "museum_env_package"

pkg_root_str = str(PKG_ROOT)
if pkg_root_str not in sys.path:
    sys.path.insert(0, pkg_root_str)
