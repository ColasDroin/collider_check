# ==================================================================================================
# --- Imports
# ==================================================================================================
# Standard library imports
import importlib.metadata

# Local imports
from .collider_check import ColliderCheck
from .utils import from_collider, from_json

# ==================================================================================================
# --- Package version
# ==================================================================================================
try:
    __version__ = importlib.metadata.version("collider-check")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "ColliderCheck",
    "from_json",
    "from_collider",
    "__version__",
]
