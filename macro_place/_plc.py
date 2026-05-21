"""
Central import point for the external PlacementCost dependency.

The TILOS MacroPlacement plc_client_os module lives in the git submodule at
external/MacroPlacement/CodeElements/Plc_client/. This module adds that path
*once* so the rest of the package can simply do:

    from macro_place._plc import PlacementCost
"""

import sys
from pathlib import Path

_PLC_CLIENT_DIR = str(
    Path(__file__).resolve().parent.parent
    / "external"
    / "MacroPlacement"
    / "CodeElements"
    / "Plc_client"
)

_PLC_CLIENT_PY = Path(_PLC_CLIENT_DIR) / "plc_client_os.py"
if not _PLC_CLIENT_PY.is_file():
    raise ImportError(
        f"plc_client_os not found at {_PLC_CLIENT_PY}.\n"
        "Initialize the MacroPlacement submodule:\n"
        "  git submodule update --init --recursive external/MacroPlacement\n"
        "Or clone manually:\n"
        "  git clone -b fix-scientific-notation-parsing "
        "https://github.com/partcleda/MacroPlacement.git external/MacroPlacement"
    )

if _PLC_CLIENT_DIR not in sys.path:
    sys.path.insert(0, _PLC_CLIENT_DIR)

from plc_client_os import PlacementCost  # noqa  # type: ignore
__all__ = ["PlacementCost"]
