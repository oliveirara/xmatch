"""
an astronomical catalogs cross-matching mini-tool

The most usual cross-mathing algorithms -- nearest neighbour and great circle --
are implemented in modules named accordingly (`nn.py` and `gc.py`); which provide
pure-positional cross-matching.
A maximum likelihood algorithm algorithm is provided in module `mle.py`, where
an extra column is necessary to be used as likelihood's *data feature*.
"""

from ._xmatchi import xmatch
from .mle import mle

__all__ = [
    "mle",
    "xmatch",
]
