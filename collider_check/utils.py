# ==================================================================================================
# --- Imports
# ==================================================================================================

# Third party imports
import xtrack as xt

# Local imports
from .collider_check import ColliderCheck

# ==================================================================================================
# --- Functions
# ==================================================================================================


def from_json(path):
    """
    Load collider_check object using a collider json file.

    Parameters
    ----------
    path : str
        Path to the collider json file.

    Returns
    -------
    collider_check : ColliderCheck
        ColliderCheck object.
    """

    collider = xt.Multiline.from_json(path)

    # Build collider_check object
    collider.build_trackers()

    return ColliderCheck(collider=collider)


def from_collider(collider):
    """
    Load collider_check object using a collider object.

    Parameters
    ----------
    collider : xtrack.Multiline
        Collider object.

    Returns
    -------
    collider_check : ColliderCheck
        ColliderCheck object.
    """

    # Build collider_check object
    try:
        collider.build_trackers()
    except Exception:
        "No trackers to build."

    return ColliderCheck(collider=collider)
