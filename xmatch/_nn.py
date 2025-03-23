import logging

from astropy.coordinates import Angle, SkyCoord, match_coordinates_sky
from numpy import append, arange, array_split

from ._utils import stats


def nn(A_coord, B_coord, parallel_process=False, nprocs=None):
    """
    Nearest-Neighbor search

    Input:
     - A_coord : ~astropy.coordinates.SkyCoord
            reference catalog (catalog "A")
     - B_coord : ~astropy.coordinates.SkyCoord
            matching catalog (catalog "B")

    Output:
     - tuple with ~numpy.ndarray , ~astropy.units.Quantity
            array of respective (to 'A_coord') index entries in 'B_coord'
            , array of respective pair distances
    """
    if parallel_process and nprocs > 0:
        # from ._utils import parallel -> TODO
        dview = None  # parallel.setup(nprocs) -> TODO
        if not dview:
            return None

        logging.debug(f"Running in parallel with {len(dview)} processors.")
        match_A_nn_idx, match_A_nn_sep = _nn_parallel(A_coord, B_coord, dview=dview)
    else:
        logging.debug("Running in serial mode.")
        match_A_nn_idx, match_A_nn_sep = _nn_serial(A_coord, B_coord)

    return match_A_nn_idx, match_A_nn_sep


def _nn_serial(A_coord, B_coord):
    if not isinstance(A_coord, SkyCoord):
        raise TypeError("Was expecting a ~astropy.coordinates.SkyCoord instance.")
    if not isinstance(B_coord, SkyCoord):
        raise TypeError("Was expecting a ~astropy.coordinates.SkyCoord instance.")

    logging.info(f"Searching among {len(B_coord)} neighbors, {len(A_coord)} reference objects.")
    logging.debug(
        f"Unit of coordinates being matched: ({A_coord.ra.unit},{A_coord.dec.unit}) and ({B_coord.ra.unit},{B_coord.dec.unit})"
    )

    match_A_nn_idx, match_A_nn_sep, _d3d = match_coordinates_sky(A_coord, B_coord)

    logging.info(f"Basic stats of distances between matchings: {stats(match_A_nn_sep.value)}")

    if len(match_A_nn_idx) != len(A_coord):
        raise ValueError("Length of match_A_nn_idx does not match length of A_coord")
    if match_A_nn_idx.max() >= len(B_coord):
        raise ValueError("Index out of bounds: match_A_nn_idx contains an index larger than the length of B_coord")

    return (match_A_nn_idx, match_A_nn_sep)


def _nn_parallel(A_coord, B_coord, dview=None):
    if not dview:
        raise ValueError("A cluster clients hub, ie. 'dview', must be given.")

    # Encapsulate some variables to send for processing
    def make_nn_search_parallel(foo, cat2):
        def pkg_nn_search(cat1, foo=foo, cat2=cat2):
            return foo(cat1, cat2)

        return pkg_nn_search

    # Split array (of coordinates) in N pieces
    def split_array(A_coord, N):
        index = arange(len(A_coord))
        A_pieces = [A_coord[idx] for idx in array_split(index, N)]
        return A_pieces

    # Join array/list of tuples in N pieces
    def join_array(A_outs):
        match_A_nn_idx = None
        match_A_nn_sep = None
        for each_out in A_outs:
            match_idx, match_sep = each_out
            if match_A_nn_idx is None:
                if match_A_nn_sep is not None:
                    raise ValueError("Expected match_A_nn_sep to be None")
                match_A_nn_idx = match_idx
                match_A_nn_sep = match_sep
            else:
                match_A_nn_idx = append(match_A_nn_idx, match_idx)
                match_A_nn_sep = append(match_A_nn_sep, match_sep)
        return (match_A_nn_idx, match_A_nn_sep)

    # A-B
    foo_match_coordinates = make_nn_search_parallel(_nn_serial, B_coord)

    A_pieces = split_array(A_coord, N=len(dview))

    A_outs = dview.map_sync(foo_match_coordinates, A_pieces)

    # This is a hack to recompose the 'unit' parameter from 'match_sep' below;
    unit_sep = A_outs[-1][-1].unit

    match_A_nn_idx, match_A_nn_sep = join_array(A_outs)

    # Do the hack: ('match_sep' loose its unit during numpy.append in 'join')
    match_A_nn_sep = Angle(match_A_nn_sep.value, unit=unit_sep)

    return (match_A_nn_idx, match_A_nn_sep)
