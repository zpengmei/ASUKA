from __future__ import annotations

"""Compatibility facade for SOC state-interaction APIs.

New implementations live in :mod:`asuka.soc.state_interaction`.
"""

from asuka.soc.state_interaction import *  # noqa: F401,F403

# Keep explicit module-level export parity for users importing from `asuka.soc.si`.
from asuka.soc.state_interaction import __all__  # type: ignore[attr-defined]
