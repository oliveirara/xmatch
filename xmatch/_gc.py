import logging

from astropy.coordinates import Angle, SkyCoord, search_around_sky

from ._utils import stats


def gc(A_coord, B_coord, radius):
    """
    Match catalogs by position within a radial distance

    Input:
     - A_coord: astropy.coordinates.SkyCoord
     - B_coord: astropy.coordinates.SkyCoord
     - radius : astropy.coordinates.Angle

    Output:
     - matched_A_indexes: indexes of A entry that match the corresponding position in B
     - matched_B_indexes: indexes of B entry that match the corresponding position in A
     - separation_AB_val: separation between matched_{AB}_indexes, astropy.coordinates.Angle

    * All outputs (1D arrays) have the same length.
    """
    if not isinstance(A_coord, SkyCoord):
        raise TypeError("Was expecting an ~astropy.coordinates.SkyCoord instance for 'A_coord'.")
    if not isinstance(B_coord, SkyCoord):
        raise TypeError("Was expecting an ~astropy.coordinates.SkyCoord instance for 'B_coord'.")

    try:
        radius = Angle(radius.arcsec, unit="arcsec")
    except AttributeError:
        radius = Angle(radius, unit="arcsec")
    if not isinstance(radius, Angle):
        raise TypeError("Was expecting an ~astropy.coordinates.Angle instance for 'radius'")

    return _gc_serial(A_coord, B_coord, radius)


def _gc_serial(A_coord, B_coord, radius):
    logging.info(f"Searching B_coord {len(B_coord)} objects, {len(A_coord)} neighbors.")

    match_A_gc_idx, match_B_gc_idx, match_gc_sep, _d3d = search_around_sky(A_coord, B_coord, radius)
    if len(match_A_gc_idx) != len(match_B_gc_idx):
        raise ValueError("The lengths of match_A_gc_idx and match_B_gc_idx do not match.")

    logging.info(f"Basic stats of distances between matchings: {stats(match_gc_sep.value)}")

    return match_A_gc_idx, match_B_gc_idx, match_gc_sep
