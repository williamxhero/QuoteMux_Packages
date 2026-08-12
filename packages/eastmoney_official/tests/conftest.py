from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if "quotemux_packages" not in sys.modules:
    spec = spec_from_file_location(
        "quotemux_packages",
        PACKAGE_ROOT / "__init__.py",
        submodule_search_locations=[str(PACKAGE_ROOT)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载本地 quotemux_packages")
    module = module_from_spec(spec)
    sys.modules["quotemux_packages"] = module
    spec.loader.exec_module(module)
