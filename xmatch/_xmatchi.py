from collections import OrderedDict
import logging

from astropy import units
from astropy.coordinates import SkyCoord
from astropy.table import Table
from numpy import arange, unique
from pandas import DataFrame, concat

from ._gc import gc
from ._nn import nn


def log(string, arg=""):
    """ """
    message = f"{string}: {arg!r}"
    logging.info(message)


class SubProducts(OrderedDict):
    """ """

    def __setitem__(self, key, value):
        logging.debug(f"Adding subproduct to stack: {value!r}")
        super().__setitem__(key, value)

    def dump(self, directory=None):
        for k, v in self.items():
            try:
                v.dump(k)
            except Exception:
                logging.exception(f"Error writing {k!s}:{v!s}")
                print(f"{k!s}:{v!s} could not be written")


products = SubProducts()


def skycoords(ra, dec, unit="degree"):
    """
    Return a ~astropy.coordinates.SkyCoord object
    """
    logging.debug(f"input::{locals()!s}")

    unit = units.Unit(unit)
    return SkyCoord(ra=ra, dec=dec, unit=unit)


def xmatch(
    catalog_A,
    catalog_B,
    columns_A=None,
    columns_B=None,
    radius=None,
    method="gc",
    separation_unit="arcsec",
    minimal=True,
    snr_column=None,
):
    """
    Returns the cross-matched catalog of same size as 'catalog_A' (left join)

    Output (table) provides the input positional columns as well as the
    measurements used during the matching process, in particular the separation
    distances between matching pairs as well as the serialized list of (eventual)
    duplicated matchings. The (physical) unit for sources separation distances is
    *arc seconds*.

    'snr_colunm' is used to filter duplicates. Say you want to remove duplicated
    matches (matches with same target but mulitiple counterparts) based on a
    a signal-to-noise (snr) measurement for each source, the 'snr_column' column
    name is to be used.

    Input:
     - catalog_{A,B} : ~pandas.DataFrame
        DFs containing (at least) the columns 'ra','dec','id'

     - columns_{A,B} : dict mapping 'ra','dec','id' columns
        In case catalog(s) have different column names for 'ra','dec','id';
        e.g, {'ra':'RA', 'dec':'Dec', 'id':'ObjID'}

     - radius: astropy' angle quantity
        Define the region size around to search for counterparts (gc only)

     - method: string
        Options are 'nn' or 'gc' (default)

     - separation_unit: string
        Unit to use in output catalog for the separation between matches;
        Options are 'arcsec', 'arcmin', 'degree'

     - minimal : bool
        If True, output catalog contains only the positional columns

     - snr_column : string
        Name of the column to use as disambiguation feature when filtering duplicates

    Output:
     - matched_catalog : ~pandas.DataFrame

    """
    parallel = False
    nprocs = None
    output_minimal = minimal

    if snr_column:
        msg = "Column named '{}' not found in ancillary catalog"
        if snr_column not in catalog_B.columns:
            raise ValueError(msg.format(snr_column))

    if not isinstance(catalog_A, DataFrame):
        raise TypeError("catalog_A must be a pandas DataFrame")
    if not isinstance(catalog_B, DataFrame):
        raise TypeError("catalog_B must be a pandas DataFrame")

    logging.debug("Arguments given", locals())

    mandatory_cols = ["ra", "dec", "id"]

    def setup_maps(default_columns, columns_map):
        """
        Input:
         - default_columns : list of column names
         - columns_map : dictionary mapping default to particular names

        Output:
         - columns_map : dictionary mapping default to particular names,
                         in case input 'columns_map' was empty (default)
                         or had no all the mappings, this output completes it.
        """
        cols_map = {c: c for c in default_columns}
        if not columns_map:
            columns_map = {}
        cols_map.update(columns_map)
        return cols_map

    catA_cols_map = setup_maps(mandatory_cols, columns_A)
    if not all(mc in catA_cols_map for mc in mandatory_cols):
        raise ValueError("Not all mandatory columns are present in catA_cols_map")

    logging.debug("Columns A map", catA_cols_map)

    catB_cols_map = setup_maps(mandatory_cols, columns_B)
    if not all(mc in catB_cols_map for mc in mandatory_cols):
        raise ValueError("Not all mandatory columns are present in catB_cols_map")

    logging.debug("Columns B map", catB_cols_map)

    def check_columns(column_names, columns_given):
        return {x for x in column_names if str(x) in columns_given} == set(columns_given)

    if not check_columns(catalog_A.columns, list(catA_cols_map.values())):
        raise ValueError(f"I looked for columns {list(catA_cols_map.values())} but could not find them in catalog_A.")
    if not check_columns(catalog_B.columns, list(catB_cols_map.values())):
        raise ValueError(f"I looked for columns {list(catB_cols_map.values())} but could not find them in catalog_B.")

    A_coord = skycoords(
        catalog_A[catA_cols_map.get("ra")].values, catalog_A[catA_cols_map.get("dec")].values, unit="degree"
    )

    B_coord = skycoords(
        catalog_B[catB_cols_map.get("ra")].values, catalog_B[catB_cols_map.get("dec")].values, unit="degree"
    )

    if method == "nn":
        match_A_nn_idx, match_A_nn_sep = nn(A_coord, B_coord, parallel_process=parallel, nprocs=nprocs)

        log(f"Total number of matchings A-B): {len(match_A_nn_idx)}")
        log(f"`- number of uniques: {len(unique(match_A_nn_idx))}")
        k = "Dist_AB"
        v = Table({"separation": match_A_nn_sep})
        products[k] = v
        log(f"Partial '{k}' product on stack", "separation between A-)B")

        match_B_nn_idx, match_B_nn_sep = nn(B_coord, A_coord, parallel_process=parallel, nprocs=nprocs)
        # tictac = timer.checkpoint('done with second matching')
        # log(tictac)

        log(f"Total number of matchings B-A): {len(match_B_nn_idx)}")
        log(f"`- number of uniques: {len(unique(match_B_nn_idx))}")
        k = "Dist_BA"
        v = Table({"separation": match_B_nn_sep})
        products[k] = v
        log(f"Partial '{k}' product on stack", "separation between B-)A")

        del match_B_nn_sep
        del A_coord, B_coord

        match_A_nn_sep = match_A_nn_sep.arcsec if separation_unit == "arcsec" else match_A_nn_sep.degree

        df_matched_idx = match_pairs(match_A_nn_idx, match_B_nn_idx, match_A_nn_sep)

    else:
        if not (method == "gc" and radius > 0):
            raise ValueError("Method must be 'gc' and radius must be greater than 0")

        match_A_gc_idx, match_B_gc_idx, match_gc_sep = gc(A_coord, B_coord, radius)

        log(f"Total number of matchings A-B): {len(match_A_gc_idx)}")
        log(f"`- number of uniques in A: {len(unique(match_A_gc_idx))}")
        log(f"`- number of uniques in B: {len(unique(match_B_gc_idx))}")

        k = "Dist_ABA"
        v = Table({"separation": match_gc_sep})
        products[k] = v
        log(f"Partial '{k}' product on stack", "separation between (A-B)")

        del A_coord, B_coord

        if snr_column is None:
            match_gc_sep = match_gc_sep.arcsec if separation_unit == "arcsec" else match_gc_sep.degree
            df_matched_idx = select_pairs(match_A_gc_idx, match_B_gc_idx, match_gc_sep)
        else:
            snr = catalog_B[snr_column].values
            snr_match = snr[match_B_gc_idx]
            df_matched_idx = select_snr(match_A_gc_idx, match_B_gc_idx, snr_match)

        del match_A_gc_idx, match_B_gc_idx, match_gc_sep

    # By this point, 'df_matched_idx' is a DataFrame containing:
    # 'A_idx' : positional indexes from table 'A'
    # 'B_idx' : positional indexes from table 'B'
    # 'separation' : angular distances between matches
    # 'duplicates' : if any, ';'-separated list of multiple matches indexes
    # 'distances'  : if any, ';'-separated list of multiple matches distances

    if output_minimal:
        catalog_A = catalog_A[list(catA_cols_map.values())]
        catalog_B = catalog_B[list(catB_cols_map.values())]

    if snr_column is not None:
        matched_catalog = merge_catalogs_snr(catalog_A, catalog_B, df_matched_idx, catB_cols_map.get("id"))
        matched_catalog.drop_duplicates([("B", "RA"), ("B", "DEC")], inplace=True)
        null_idx = matched_catalog.loc[matched_catalog["B", "OBJID"].isnull()]
        if len(null_idx):
            matched_catalog.drop(null_idx.index, inplace=True)
    else:
        matched_catalog = merge_catalogs(catalog_A, catalog_B, df_matched_idx, catB_cols_map.get("id"))

    return matched_catalog


def merge_catalogs_snr(A, B, df_matched_idx, B_id_column=None):
    """
    Tables 'A' and 'B' are 'left' joined according to 'df_matched_idx'

    'df_matched_idx' is a table with columns:
    - 'A_idx', representing the primary-key for catalog 'A'
    - 'B_idx', representing the primary-key for catalog 'B'
    - 'separation', distances between matches
    - 'duplicates' (optional), the ';'-separated list of multiple matches
    - 'distances' (optional), the ';'-separated list of duplicates' separation

    If 'B_id_column' is given, it should be the name of object IDs column in 'B'

    Return the joined from A,B and separation provided by 'df_matched_idx'
    """

    AB_match = DataFrame({"snr": df_matched_idx["snr"].values}, index=df_matched_idx["A_idx"].values)

    if "duplicates" in df_matched_idx.columns:
        if "snrs" not in df_matched_idx.columns:
            raise ValueError("'snrs' column not found in df_matched_idx")

        if B_id_column in B.columns:
            for i, row in df_matched_idx.iterrows():
                entry_ids = row["duplicates"]
                entry_sep = row["snrs"]
                # print('ENTRIES: "{}" , "{}"'.format(entry_ids, entry_sep))
                if entry_ids is None or len(entry_ids) == 0 or entry_ids.lower() == "none":
                    if entry_sep != entry_ids:
                        raise ValueError(f"Distances '{entry_sep}' expected to be None")
                    continue
                # print('CONTINS: "{}" , "{}"'.format(entry_ids, entry_sep))

                # entry_ids are the former position (row number) in table
                inds = [int(n) for n in entry_ids.split(";")]
                ids = ";".join([str(n) for n in B[B_id_column].iloc[inds]])

                if len(ids.split(";")) != len(entry_sep.split(";")):
                    raise ValueError("Mismatch between ids and entry_sep lengths")

                df_matched_idx.loc[i, "duplicates"] = ids
                df_matched_idx.loc[i, "snrs"] = entry_sep

        AB_match.loc[:, "duplicates"] = df_matched_idx["duplicates"].values
        AB_match.loc[:, "snrs"] = df_matched_idx["snrs"].values

    B_matched = B.iloc[df_matched_idx["B_idx"]]
    B_matched.loc[:, "A_idx"] = df_matched_idx["A_idx"].values

    B_matched = B_matched.set_index("A_idx")

    df = concat([A, B_matched, AB_match], axis=1, keys=["A", "B", "AB"])

    return df


def select_snr(match_A_gc_idx, match_B_gc_idx, snr):
    """
    Return table with 'match_A_gc_idx' unique values as index

    The output table contains five columns:
    - 'A_idx', 'match_A_gc_idx' unique values
    - 'B_idx', 'match_B_gc_idx' value of the corresponding nearest match
    - 'separation': distance between the nearest A-B matches
    - 'duplicates': if multiple 'A_idx', the corresponding 'B_idx' values
    - 'distances' : if multiple 'A_idx', the corresponding 'separation' values
    """

    df = DataFrame({"A_idx": match_A_gc_idx, "B_idx": match_B_gc_idx, "snr": snr, "duplicates": None, "snrs": None})

    primary_entries = []
    duplicated_idx = set()
    for gname, gdf in df.groupby("A_idx"):
        if gname in duplicated_idx:
            # print('GNAME already out', gname)
            # dl = list(duplicated_idx)
            # dl.sort()
            # print('Duplicated', dl)
            # raise
            continue

        if len(gdf) == 1:
            # Notice that the "index" going to 'primary' is df's index
            primary_entries.append(gdf.index[0])
            continue

        to_keep = gdf["snr"] == gdf["snr"].max()

        msg = f"Indexes differ: {to_keep},{gdf}"
        if not all(to_keep.index == gdf.index):
            raise ValueError(msg)

        if sum(to_keep) != 1:
            if not (1 < sum(to_keep) <= len(gdf)):
                raise ValueError(f"to-KEEP size error:\n{to_keep}\n{gdf}")
            # print("VARIOUS to-KEEP:\n{}\n".format(to_keep))

            # Notice that 'ikeep' is the index of 'df'
            ikeep = to_keep[to_keep].index[0]
            to_keep = to_keep * False
            to_keep.loc[ikeep] = True

        to_drop = ~to_keep

        if not all(to_drop.index == gdf.index):
            raise ValueError("Indexes do not match")
        if not all(to_keep.index == gdf.index):
            raise ValueError("Indexes do not match")

        to_drop_i = to_drop[to_drop].index
        to_keep_i = to_keep[to_keep].index

        if len(to_drop_i) != sum(to_drop):
            raise ValueError("Mismatch between length of to_drop_i and sum of to_drop")
        if len(to_keep_i) != sum(to_keep):
            raise ValueError("Mismatch between length of to_keep_i and sum of to_keep")
        # print("\n{}\n{}\n".format(to_drop, to_keep))
        # print("\n{}\n{}\n".format(to_drop_i, to_keep_i))

        # print("TO-KEEP/DROP:", to_keep, to_drop)
        # print('{}\n{}'.format(gname,gdf))

        if len(to_keep_i) != 1:
            raise ValueError(f"found more than one primary objects?: {to_keep}")
        if sum(to_drop) + sum(to_keep) != len(gdf):
            raise ValueError(f"to_drop: {to_drop}\nto_keep: {to_keep}")

        to_keep_i = to_keep_i[0]

        # print('gname',gname)
        # print('to-drop:\n',gdf.loc[to_drop, 'B_idx'])
        # print('to-keep:\n',gdf.loc[to_keep, 'B_idx'])
        # print()
        # entries_to_drop = df.loc[to_drop]
        # B_idx_duplicated = entries_to_drop['B_idx'].values  #.tolist()
        B_idx_duplicated = gdf.loc[to_drop, "B_idx"].values
        df.loc[to_keep_i, "duplicates"] = ";".join([str(i) for i in B_idx_duplicated])

        B_sep_duplicated = gdf.loc[to_drop, "snr"].values
        df.loc[to_keep_i, "snrs"] = ";".join([str(i) for i in B_sep_duplicated])

        duplicated_idx = duplicated_idx.union(B_idx_duplicated.tolist())

        # assert all(bi not in duplicated_idx for bi in B_idx_duplicated)
        # msg = "Index to keep: '{}', indexes to drop: '{}', B_idx dup: {}".format(to_keep_i, to_drop_i, B_idx_duplicated)
        # assert to_keep_i not in B_idx_duplicated, msg

        primary_entries.append(to_keep_i)

    # Notice that 'B_idx' may still have duplicates, only 'A_idx' was cleaned from duplicates!
    # idx_to_drop = df['A_idx'].isin(duplicated_idx)
    # print("Size of df:",len(df))
    # print("Size of dup indexes:",len(duplicated_idx))
    # print("Size of indexes to keep:",len(primary_entries))
    df = df.loc[primary_entries]
    # print(df)

    # dl = list(duplicated_idx)
    # dl.sort()
    # print('Duplicated', dl)
    #
    # print(df)
    return df


def merge_catalogs(A, B, df_matched_idx, B_id_column=None):
    """
    Tables 'A' and 'B' are 'left' joined according to 'df_matched_idx'

    'df_matched_idx' is a table with columns:
    - 'A_idx', representing the primary-key for catalog 'A'
    - 'B_idx', representing the primary-key for catalog 'B'
    - 'separation', distances between matches
    - 'duplicates' (optional), the ';'-separated list of multiple matches
    - 'distances' (optional), the ';'-separated list of duplicates' separation

    If 'B_id_column' is given, it should be the name of object IDs column in 'B'

    Return the joined from A,B and separation provided by 'df_matched_idx'
    """

    AB_match = DataFrame({"separation": df_matched_idx["separation"].values}, index=df_matched_idx["A_idx"].values)

    if "duplicates" in df_matched_idx.columns:
        if "distances" not in df_matched_idx.columns:
            raise ValueError("'distances' column not found in df_matched_idx")

        if B_id_column in B.columns:
            for i, row in df_matched_idx.iterrows():
                entry_ids = row["duplicates"]
                entry_sep = row["distances"]

                if entry_ids is None:
                    if entry_sep is not None:
                        raise ValueError(f"Distances '{entry_sep}' expected to be None")
                    continue

                # entry_ids are the former position (row number) in table
                inds = [int(n) for n in entry_ids.split(";")]
                ids = ";".join([str(n) for n in B[B_id_column].iloc[inds]])

                if len(ids.split(";")) != len(entry_sep.split(";")):
                    raise ValueError("Mismatch between ids and entry_sep lengths")

                df_matched_idx.loc[i, "duplicates"] = ids
                df_matched_idx.loc[i, "distances"] = entry_sep

        AB_match.loc[:, "duplicates"] = df_matched_idx["duplicates"].values
        AB_match.loc[:, "distances"] = df_matched_idx["distances"].values

    B_matched = B.iloc[df_matched_idx["B_idx"]]
    B_matched.loc[:, "A_idx"] = df_matched_idx["A_idx"].values

    B_matched = B_matched.set_index("A_idx")

    df = concat([A, B_matched, AB_match], axis=1, keys=["A", "B", "AB"])

    return df


def select_pairs(match_A_gc_idx, match_B_gc_idx, match_gc_sep):
    """
    Return table with 'match_A_gc_idx' unique values as index

    The output table contains five columns:
    - 'A_idx', 'match_A_gc_idx' unique values
    - 'B_idx', 'match_B_gc_idx' value of the corresponding nearest match
    - 'separation': distance between the nearest A-B matches
    - 'duplicates': if multiple 'A_idx', the corresponding 'B_idx' values
    - 'distances' : if multiple 'A_idx', the corresponding 'separation' values
    """

    df = DataFrame({
        "A_idx": match_A_gc_idx,
        "B_idx": match_B_gc_idx,
        "separation": match_gc_sep,
        "duplicates": None,
        "distances": None,
    })

    idx_to_drop = []
    for _, gdf in df.groupby("A_idx"):
        if len(gdf) == 1:
            continue

        # to_drop = gdf['separation'] > gdf['separation'].min()

        to_keep = gdf["separation"] == gdf["separation"].min()

        msg = f"Indexes differ: {to_keep},{gdf}"
        if not all(to_keep.index == gdf.index):
            raise ValueError(msg)

        if sum(to_keep) != 1:
            if not (1 < sum(to_keep) <= len(gdf)):
                raise ValueError(f"to-KEEP size error:\n{to_keep}\n{gdf}")

            # Notice that 'ikeep' is the index of 'df'
            ikeep = to_keep[to_keep].index[0]
            to_keep = to_keep * False
            to_keep.loc[ikeep] = True

        to_drop = ~to_keep

        if not all(to_drop.index == gdf.index):
            raise ValueError("Indexes do not match")
        if not all(to_keep.index == gdf.index):
            raise ValueError("Indexes do not match")

        to_drop_i = to_drop[to_drop].index
        to_keep_i = to_keep[to_keep].index

        if len(to_drop_i) != sum(to_drop):
            raise ValueError("Mismatch between length of to_drop_i and sum of to_drop")
        if len(to_keep_i) != sum(to_keep):
            raise ValueError("Mismatch between length of to_keep_i and sum of to_keep")
        #        to_drop = gdf['separation'] > gdf['separation'].min()
        #        assert sum(to_drop) >= 1, "There should be at least one entry to drop; {!s}".format(gdf)
        #
        #        to_keep = to_drop[~to_drop].index
        #        to_drop = to_drop[to_drop].index

        entries_to_drop = gdf.loc[to_drop]
        B_idx_duplicated = entries_to_drop["B_idx"].values.tolist()
        df.loc[to_keep_i, "duplicates"] = ";".join([str(i) for i in B_idx_duplicated])
        B_sep_duplicated = entries_to_drop["separation"].values.tolist()
        df.loc[to_keep_i, "distances"] = ";".join([str(i) for i in B_sep_duplicated])
        del entries_to_drop

        idx_to_drop.extend(list(to_drop_i))

    # Notice that 'B_idx' may still have duplicates, only 'A_idx' was cleaned from duplicates!
    df.drop(idx_to_drop, inplace=True)

    return df


def match_pairs(match_A_nn_idx, match_B_nn_idx, match_A_nn_sep):
    """
    Remove degenerated matches from given lists

    'match_A_nn_idx' and 'match_B_nn_idx' may have duplicated values,
    meaning that the NearestNeighbor algorithm (in both directions)
    found different matches for each direction. This function pick
    one matching pair only.
    """

    A_matched_pairs = list(zip(arange(len(match_A_nn_idx)), match_A_nn_idx, strict=True))
    B_matched_pairs = set(zip(match_B_nn_idx, arange(len(match_B_nn_idx)), strict=True))

    matched_pairs = []
    matched_dists = []
    for i, p in enumerate(A_matched_pairs):
        if p in B_matched_pairs:
            matched_pairs.append(p)
            matched_dists.append(match_A_nn_sep[i])

    A_matched_idx, B_matched_idx = list(zip(*matched_pairs, strict=True))

    if len(A_matched_idx) != len(B_matched_idx):
        raise ValueError("Length of A_matched_idx does not match length of B_matched_idx")
    if len(A_matched_idx) != len(matched_dists):
        raise ValueError("Length of A_matched_idx does not match length of matched_dists")

    import pandas

    df_matched_idx = pandas.DataFrame({"A_idx": A_matched_idx, "B_idx": B_matched_idx, "separation": matched_dists})
    # df_matched_idx = df_matched_idx.set_index('A_idx')
    return df_matched_idx
