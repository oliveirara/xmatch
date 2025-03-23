from collections import OrderedDict
import logging

from numpy import amax, amin, mean, percentile, std


def stats(arrei):
    """
    Outputs in a dictionary:
    - min
    - max
    - mean
    - std
    - quantiles (1,2,3)
    """
    if not len(arrei):
        logging.error("There is no data to comute stats; given array is empty.")
        return None

    sts = OrderedDict()

    sts["length"] = len(arrei)

    if all(hasattr(arrei, attr) for attr in ["min", "max", "mean", "std"]):
        sts["min"] = arrei.min()
        sts["max"] = arrei.max()
        sts["mean"] = arrei.mean()
        sts["std"] = arrei.std()
    else:
        sts["min"] = amin(arrei)
        sts["max"] = amax(arrei)
        sts["mean"] = mean(arrei)
        sts["std"] = std(arrei)

    q1 = percentile(arrei, 25)
    q2 = percentile(arrei, 50)
    q3 = percentile(arrei, 75)
    sts["25%"] = q1
    sts["50%"] = q2
    sts["75%"] = q3

    return sts
