import os
import re
import sys
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm

try:
    import orjson
    def _parse(line):
        return orjson.loads(line)
except ImportError:
    import json
    def _parse(line):
        return json.loads(line)
    print("Tip: `pip install orjson` for 3-5x faster JSON parsing", flush=True)

K = 20
N_WORKERS = mp.cpu_count()
FILEPATH = "HotelRec.txt"

SUBRATING_KEYS = ["service", "cleanliness", "value", "rooms", "location", "sleep quality"]

GEO_RE = re.compile(rb"-g(\d+)-")

FEATURES = ["cf", "temporal", "subratings", "user_mean", "item_mean", "user_count", "item_count",
            "location", "user_dev", "tfidf", "user_subratings", "user_var", "hotel_name",
            "item_ema", "user_ema", "date_numeric"]
EMA_TAUS_DAYS = [180.0, 730.0]  # short (~6mo) and long (~2yr) half-lives
METHODS = ["knn", "lightgbm"]
DEFAULT_FEATURES = {
    "knn": ["cf", "temporal"],
    "lightgbm": list(FEATURES),
}


def parse_chunk(args):
    filepath, start_byte, end_byte = args
    dates, users, items, ratings, geos = [], [], [], [], []
    sub = [[] for _ in SUBRATING_KEYS]
    nan = float("nan")
    with open(filepath, "rb") as f:
        if start_byte > 0:
            f.seek(start_byte)
            f.readline()  # skip partial first line
        while f.tell() < end_byte:
            line = f.readline()
            if not line:
                break
            obj = _parse(line)
            dates.append(obj["date"])
            users.append(obj["author"])
            url = obj["hotel_url"]
            items.append(url)
            ratings.append(obj["rating"])
            m = GEO_RE.search(url.encode("ascii", "ignore") if isinstance(url, str) else url)
            geos.append(int(m.group(1)) if m else -1)
            pd_dict = obj.get("property_dict") or {}
            for k, col in zip(SUBRATING_KEYS, sub):
                v = pd_dict.get(k)
                col.append(float(v) if v is not None else nan)
    return dates, users, items, ratings, sub, geos


_progress = None  # shared counter, set before fork

def predict_chunk_fork(args):
    start, end = args
    preds = np.full(end - start, np.nan, dtype=np.float64)

    for idx in range(start, end):
        local = idx - start
        u = _val_u_idx[idx]
        i = _val_i_idx[idx]

        # Cold start: use precomputed fallback (user mean or global mean)
        if u < 0 or i < 0:
            preds[local] = _val_cold_start_preds[idx]
            if _progress is not None and (local + 1) % 1000 == 0:
                _progress.value += 1000
            continue

        q = _item_to_query[i]

        neighbor_dists = _distances_all[q]
        neighbor_items = _indices_all[q]

        u_items = _user_item_indices[u]
        u_ratings = _user_item_ratings[u]

        mask = neighbor_items != i
        nb_items = neighbor_items[mask][:_K]
        cf_sims = 1.0 - neighbor_dists[mask][:_K]

        # Blend in temporal similarity (1 - normalized euclidean distance)
        if _temporal_weight > 0:
            t_query = _item_temporal[i]
            t_neighbors = _item_temporal[nb_items]
            t_dists = np.linalg.norm(t_neighbors - t_query, axis=1)
            t_sims = 1.0 - t_dists / (np.sqrt(_item_temporal.shape[1] * 4) + 1e-9)  # max dist for n sin/cos features in [-1,1]
            nb_sims = (1.0 - _temporal_weight) * cf_sims + _temporal_weight * t_sims
        else:
            nb_sims = cf_sims

        if len(u_items) == 0:
            preds[local] = _user_means[u]
            if _progress is not None and (local + 1) % 1000 == 0:
                _progress.value += 1000
            continue

        pos = np.searchsorted(u_items, nb_items)
        pos = np.clip(pos, 0, len(u_items) - 1)
        found = u_items[pos] == nb_items
        valid = found & (nb_sims > 0)

        if valid.any():
            sims = nb_sims[valid]
            rats = u_ratings[pos[valid]]
            preds[local] = np.dot(sims, rats) / sims.sum()
        else:
            preds[local] = _user_means[u]

        if _progress is not None and (local + 1) % 1000 == 0:
            _progress.value += 1000

    if _progress is not None:
        _progress.value += (end - start) % 1000 or 0

    return preds


def baseline():
    """Predict global mean for all validation samples."""
    print("Loading data (single-threaded for baseline)...", flush=True)
    t0 = time.time()

    dates = []
    ratings = []
    with open(FILEPATH, "r") as f:
        for i, line in enumerate(f):
            obj = _parse(line)
            dates.append(obj["date"])
            ratings.append(obj["rating"])
            if (i + 1) % 5_000_000 == 0:
                print(f"  loaded {i+1:,} lines ({time.time()-t0:.0f}s)", flush=True)

    n = len(dates)
    print(f"  {n:,} records loaded ({time.time()-t0:.0f}s)")

    print("Sorting by date...", flush=True)
    order = np.argsort(dates, kind="mergesort")
    ratings_arr = np.array([ratings[j] for j in order], dtype=np.float32)
    del dates, ratings, order

    train_end = int(n * 0.90)
    val_end = int(n * 0.95)
    global_mean = float(ratings_arr[:train_end].mean())

    val_ratings = ratings_arr[train_end:val_end]
    y_pred = np.full(len(val_ratings), global_mean, dtype=np.float32)

    print(f"\nBaseline (predict global mean = {global_mean:.3f}):")
    print(f"  Val samples: {len(val_ratings):,}")
    print(f"  MAE:  {mean_absolute_error(val_ratings, y_pred):.4f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(val_ratings, y_pred)):.4f}")
    print(f"  Total time: {time.time()-t0:.0f}s")


def _load_sorted():
    """Parallel load + sort by date. Returns (n, dates, users, items, ratings_arr, sub_arr, t0)."""
    print(f"Loading data ({N_WORKERS} workers)...", flush=True)
    t0 = time.time()

    file_size = os.path.getsize(FILEPATH)
    boundaries = [i * file_size // N_WORKERS for i in range(N_WORKERS)] + [file_size]
    chunks = [(FILEPATH, boundaries[i], boundaries[i + 1]) for i in range(N_WORKERS)]

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        results = list(pool.map(parse_chunk, chunks))

    dates, users, items, ratings, geos = [], [], [], [], []
    sub = [[] for _ in SUBRATING_KEYS]
    for d, u, it, r, s, g in results:
        dates.extend(d)
        users.extend(u)
        items.extend(it)
        ratings.extend(r)
        geos.extend(g)
        for k_idx, col in enumerate(s):
            sub[k_idx].extend(col)
    del results

    n = len(dates)
    print(f"  {n:,} records loaded ({time.time()-t0:.0f}s)")

    print("Sorting by date...", flush=True)
    order = np.argsort(dates, kind="mergesort")
    dates = [dates[j] for j in order]
    users = [users[j] for j in order]
    items = [items[j] for j in order]
    ratings_arr = np.array([ratings[j] for j in order], dtype=np.float32)
    geos_arr = np.asarray(geos, dtype=np.int64)[order]
    sub_arr = np.empty((n, len(SUBRATING_KEYS)), dtype=np.float32)
    for k_idx in range(len(SUBRATING_KEYS)):
        sub_arr[:, k_idx] = np.asarray(sub[k_idx], dtype=np.float32)[order]
    del ratings, sub, geos, order

    return n, dates, users, items, ratings_arr, sub_arr, geos_arr, t0


def _temporal_features(dates):
    """Returns (cyclical sin/cos features for doy, numeric days since 2000-01-01)."""
    dt = pd.to_datetime(dates)
    doy = dt.dayofyear.values.astype(np.float32)
    two_pi = 2.0 * np.pi
    feats = np.column_stack([
        np.sin(two_pi * doy / 365.0),
        np.cos(two_pi * doy / 365.0),
    ]).astype(np.float32)
    date_numeric = ((dt.values - np.datetime64("2000-01-01")) / np.timedelta64(1, "D")).astype(np.float32)
    return feats, date_numeric


def _build_indices(users, items, end):
    user_to_idx, item_to_idx = {}, {}
    for j in range(end):
        u, it = users[j], items[j]
        if u not in user_to_idx:
            user_to_idx[u] = len(user_to_idx)
        if it not in item_to_idx:
            item_to_idx[it] = len(item_to_idx)
    return user_to_idx, item_to_idx


def _build_item_user_matrix(users, items, ratings_arr, user_to_idx, item_to_idx, end):
    row_idx = np.empty(end, dtype=np.int32)
    col_idx = np.empty(end, dtype=np.int32)
    for j in range(end):
        row_idx[j] = item_to_idx[items[j]]
        col_idx[j] = user_to_idx[users[j]]

    pair_rating = {}
    for j in range(end):
        pair_rating[(row_idx[j], col_idx[j])] = ratings_arr[j]

    dedup_rows = np.array([k[0] for k in pair_rating], dtype=np.int32)
    dedup_cols = np.array([k[1] for k in pair_rating], dtype=np.int32)
    dedup_data = np.array(list(pair_rating.values()), dtype=np.float32)
    print(f"  {end:,} -> {len(dedup_data):,} after dedup")

    n_items = len(item_to_idx)
    n_users = len(user_to_idx)
    return csr_matrix(
        (dedup_data, (dedup_rows, dedup_cols)),
        shape=(n_items, n_users), dtype=np.float32,
    )


def _per_item_temporal(items, item_to_idx, temporal_feats, end):
    n_items = len(item_to_idx)
    n_temporal = temporal_feats.shape[1]
    item_temporal_sum = np.zeros((n_items, n_temporal), dtype=np.float64)
    item_count = np.zeros(n_items, dtype=np.int32)
    for j in range(end):
        it = items[j]
        if it in item_to_idx:
            idx = item_to_idx[it]
            item_temporal_sum[idx] += temporal_feats[j]
            item_count[idx] += 1
    item_count_safe = np.maximum(item_count, 1).reshape(-1, 1)
    return (item_temporal_sum / item_count_safe).astype(np.float32)


def _user_item_lookup(item_user_mat, n_users, global_mean, shrinkage_lambda=10):
    csc = item_user_mat.tocsc()
    user_item_indices = []
    user_item_ratings = []
    user_means = np.full(n_users, global_mean, dtype=np.float32)
    user_counts = np.zeros(n_users, dtype=np.int32)
    for u in range(n_users):
        s, e = csc.indptr[u], csc.indptr[u + 1]
        user_item_indices.append(csc.indices[s:e].copy())
        user_item_ratings.append(csc.data[s:e].copy())
        n_i = e - s
        user_counts[u] = n_i
        if n_i > 0:
            raw_mean = csc.data[s:e].mean()
            user_means[u] = (n_i * raw_mean + shrinkage_lambda * global_mean) / (n_i + shrinkage_lambda)
    return user_item_indices, user_item_ratings, user_means, user_counts


def _index_arrays(users, items, user_to_idx, item_to_idx, start, end):
    """Per-row (u_idx, i_idx) for slice [start, end), -1 for unknown."""
    n = end - start
    u_idx = np.empty(n, dtype=np.int32)
    i_idx = np.empty(n, dtype=np.int32)
    for k, j in enumerate(range(start, end)):
        u_idx[k] = user_to_idx.get(users[j], -1)
        i_idx[k] = item_to_idx.get(items[j], -1)
    return u_idx, i_idx


def _build_geo_index(geos_arr, end):
    """Build geo_to_idx mapping from feat-train slice; return (geo_to_idx, geo_idx_for_slice)."""
    geo_to_idx = {}
    geo_idx = np.full(end, -1, dtype=np.int32)
    for j in range(end):
        g = int(geos_arr[j])
        if g >= 0:
            if g not in geo_to_idx:
                geo_to_idx[g] = len(geo_to_idx)
            geo_idx[j] = geo_to_idx[g]
    return geo_to_idx, geo_idx


def _geo_idx_for_slice(geos_arr, geo_to_idx, start, end):
    """Look up geo indices for an existing mapping over [start, end). -1 for cold geos."""
    out = np.full(end - start, -1, dtype=np.int32)
    for k, j in enumerate(range(start, end)):
        g = int(geos_arr[j])
        if g >= 0 and g in geo_to_idx:
            out[k] = geo_to_idx[g]
    return out


def _region_aggregates(geo_idx, ratings, n_regions, global_mean, shrinkage_lambda=10):
    """Shrunk per-region mean and per-region rating count over the feat-train slice."""
    region_sum = np.zeros(n_regions, dtype=np.float64)
    region_count = np.zeros(n_regions, dtype=np.int64)
    valid = geo_idx >= 0
    np.add.at(region_sum, geo_idx[valid], ratings[valid].astype(np.float64))
    np.add.at(region_count, geo_idx[valid], 1)
    region_mean = ((region_sum + shrinkage_lambda * global_mean) /
                   (region_count + shrinkage_lambda)).astype(np.float32)
    return region_mean, region_count


def _prior_sum_count(idx, values, valid_value_mask=None):
    """For each row j with valid idx[j], compute (prior_sum, prior_count) over rows k with idx[k]==idx[j]
    and date[k] < date[j]. Assumes input arrays are already date-sorted globally.

    valid_value_mask: optional bool array; rows where False don't contribute to sum/count
    (e.g., NaN sub-rating values). Applies in addition to idx >= 0."""
    n = len(idx)
    idx_valid = idx >= 0
    if valid_value_mask is None:
        contributes = idx_valid
    else:
        contributes = idx_valid & valid_value_mask

    val_for_sum = np.where(contributes, values, 0.0).astype(np.float64)
    val_for_count = contributes.astype(np.float64)

    # Stable sort by idx; preserves global date order within each group
    order = np.argsort(idx, kind="stable")
    idx_sorted = idx[order]
    val_sum_sorted = val_for_sum[order]
    val_cnt_sorted = val_for_count[order]

    # Group boundaries
    group_changes = np.empty(n, dtype=bool)
    group_changes[0] = True
    if n > 1:
        group_changes[1:] = idx_sorted[1:] != idx_sorted[:-1]
    start_pos = np.where(group_changes, np.arange(n), 0).astype(np.int64)
    np.maximum.accumulate(start_pos, out=start_pos)

    # Cumulative sums and "base" (cumsum just before group start)
    cum_sum_global = np.cumsum(val_sum_sorted)
    cum_cnt_global = np.cumsum(val_cnt_sorted)
    base_sum = np.where(start_pos > 0, cum_sum_global[np.maximum(start_pos - 1, 0)], 0.0)
    base_cnt = np.where(start_pos > 0, cum_cnt_global[np.maximum(start_pos - 1, 0)], 0.0)

    # Within-group cumulative INCLUSIVE of current row
    cum_sum_within = cum_sum_global - base_sum
    cum_cnt_within = cum_cnt_global - base_cnt

    # Prior = inclusive minus current row's contribution
    prior_sum_sorted = cum_sum_within - val_sum_sorted
    prior_cnt_sorted = cum_cnt_within - val_cnt_sorted

    prior_sum = np.empty(n, dtype=np.float64)
    prior_cnt = np.empty(n, dtype=np.float64)
    prior_sum[order] = prior_sum_sorted
    prior_cnt[order] = prior_cnt_sorted

    # Zero out rows with invalid idx (no group)
    prior_sum[~idx_valid] = 0.0
    prior_cnt[~idx_valid] = 0.0

    return prior_sum, prior_cnt.astype(np.int64)


def _prior_sum_sqsum_count(idx, values):
    """Like _prior_sum_count but also returns prior cumulative sum-of-squares (for variance)."""
    n = len(idx)
    idx_valid = idx >= 0
    val_sum = np.where(idx_valid, values, 0.0).astype(np.float64)
    val_sq = val_sum * val_sum
    val_cnt = idx_valid.astype(np.float64)

    order = np.argsort(idx, kind="stable")
    idx_sorted = idx[order]
    s_sorted = val_sum[order]
    sq_sorted = val_sq[order]
    c_sorted = val_cnt[order]

    group_changes = np.empty(n, dtype=bool)
    group_changes[0] = True
    if n > 1:
        group_changes[1:] = idx_sorted[1:] != idx_sorted[:-1]
    start_pos = np.where(group_changes, np.arange(n), 0).astype(np.int64)
    np.maximum.accumulate(start_pos, out=start_pos)

    cs = np.cumsum(s_sorted)
    csq = np.cumsum(sq_sorted)
    cc = np.cumsum(c_sorted)
    bs = np.where(start_pos > 0, cs[np.maximum(start_pos - 1, 0)], 0.0)
    bsq = np.where(start_pos > 0, csq[np.maximum(start_pos - 1, 0)], 0.0)
    bc = np.where(start_pos > 0, cc[np.maximum(start_pos - 1, 0)], 0.0)

    prior_sum_s = (cs - bs) - s_sorted
    prior_sq_s = (csq - bsq) - sq_sorted
    prior_cnt_s = (cc - bc) - c_sorted

    prior_sum = np.empty(n, dtype=np.float64)
    prior_sq = np.empty(n, dtype=np.float64)
    prior_cnt = np.empty(n, dtype=np.float64)
    prior_sum[order] = prior_sum_s
    prior_sq[order] = prior_sq_s
    prior_cnt[order] = prior_cnt_s

    prior_sum[~idx_valid] = 0.0
    prior_sq[~idx_valid] = 0.0
    prior_cnt[~idx_valid] = 0.0
    return prior_sum, prior_sq, prior_cnt.astype(np.int64)


def _ema_aggregates(idx, ratings_slice, date_numeric_slice, threshold_date_num, n_groups, taus):
    """Time-decayed weighted-average rating per group, evaluated at threshold_date_num.
    Returns (out_emas, weight_totals, wr_totals): each (n_groups, n_taus). NaN for groups with no ratings."""
    valid = idx >= 0
    days_back = (threshold_date_num - date_numeric_slice).astype(np.float64)
    n_taus = len(taus)
    out = np.full((n_groups, n_taus), np.nan, dtype=np.float32)
    weight_totals = np.zeros((n_groups, n_taus), dtype=np.float64)
    wr_totals = np.zeros((n_groups, n_taus), dtype=np.float64)
    ratings_f64 = ratings_slice.astype(np.float64)
    for k, tau in enumerate(taus):
        weights = np.exp(-days_back / float(tau))
        wr = weights * ratings_f64
        wsum = np.zeros(n_groups, dtype=np.float64)
        rsum = np.zeros(n_groups, dtype=np.float64)
        np.add.at(wsum, idx[valid], weights[valid])
        np.add.at(rsum, idx[valid], wr[valid])
        weight_totals[:, k] = wsum
        wr_totals[:, k] = rsum
        out[:, k] = np.where(wsum > 0, rsum / np.maximum(wsum, 1e-12), np.nan).astype(np.float32)
    return out, weight_totals, wr_totals


def _user_rating_variance_aggregates(u_idx, ratings, n_users):
    """Per-user rating sum, sum-of-squares, and count (used for variance)."""
    valid_u = u_idx >= 0
    ratings_f64 = ratings.astype(np.float64)
    user_sum = np.zeros(n_users, dtype=np.float64)
    user_sqsum = np.zeros(n_users, dtype=np.float64)
    user_count = np.zeros(n_users, dtype=np.int64)
    np.add.at(user_sum, u_idx[valid_u], ratings_f64[valid_u])
    np.add.at(user_sqsum, u_idx[valid_u], ratings_f64[valid_u] ** 2)
    np.add.at(user_count, u_idx[valid_u], 1)
    return user_sum, user_sqsum, user_count


def _hotel_name_embeddings(item_urls, n_components=16, max_features=2000):
    """TF-IDF over hotel name+location strings parsed from URLs, then TruncatedSVD."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    name_strs = []
    for url in item_urls:
        if url is None:
            name_strs.append("")
            continue
        s = url[:-5] if url.endswith(".html") else url
        s = s.replace("_", " ").replace("-", " ").lower()
        name_strs.append(s)
    vec = TfidfVectorizer(min_df=5, max_features=max_features,
                          token_pattern=r"\b[a-z][a-z]{2,}\b")
    X = vec.fit_transform(name_strs)
    n_components = min(n_components, X.shape[1] - 1) if X.shape[1] > 1 else 1
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    embeddings = svd.fit_transform(X).astype(np.float32)
    return embeddings, vec, svd.explained_variance_ratio_.sum()


def _user_subrating_aggregates(u_idx, sub_slice, n_users):
    """Per-user sub-rating sums and counts per key. Returns (sums, counts, per-key global means)."""
    n_keys = sub_slice.shape[1]
    valid_u = u_idx >= 0
    sums = np.zeros((n_users, n_keys), dtype=np.float64)
    counts = np.zeros((n_users, n_keys), dtype=np.int64)
    for k in range(n_keys):
        v = sub_slice[:, k]
        m = ~np.isnan(v) & valid_u
        np.add.at(sums[:, k], u_idx[m], v[m].astype(np.float64))
        np.add.at(counts[:, k], u_idx[m], 1)
    global_count = counts.sum(axis=0).astype(np.float64)
    global_sum = sums.sum(axis=0)
    global_means = np.where(global_count > 0, global_sum / np.maximum(global_count, 1), 0.0)
    return sums, counts, global_means


def _user_dev_aggregates(u_idx, i_idx, ratings, item_means, n_users, shrinkage_lambda=10):
    """Per-user shrunk mean of (rating - item_mean[i]). Captures critic-vs-generous bias."""
    valid_u = u_idx >= 0
    valid_i = i_idx >= 0
    valid = valid_u & valid_i
    safe_i = np.where(valid_i, i_idx, 0)
    dev = ratings.astype(np.float64) - item_means[safe_i].astype(np.float64)
    dev_sum = np.zeros(n_users, dtype=np.float64)
    dev_count = np.zeros(n_users, dtype=np.int64)
    np.add.at(dev_sum, u_idx[valid], dev[valid])
    np.add.at(dev_count, u_idx[valid], 1)
    user_dev = np.where(dev_count > 0,
                        dev_sum / (dev_count + shrinkage_lambda),
                        np.nan).astype(np.float32)
    return user_dev, dev_count


def _hotel_tfidf_embeddings(filepath, item_to_idx, n_items, threshold_date,
                             hash_dim=8192, n_components=32):
    """Stream-pass over file: per-hotel hashed token counts → IDF → L2-normalize → TruncatedSVD."""
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.decomposition import TruncatedSVD

    print(f"  Building hotel TF-IDF (hash_dim={hash_dim}, dims={n_components})...", flush=True)
    t0 = time.time()
    vec = HashingVectorizer(
        n_features=hash_dim, alternate_sign=False, norm=None, lowercase=True,
        token_pattern=r"(?u)\b[a-z][a-z]{2,}\b",
    )

    BATCH = 20_000
    batch_hotels = []
    batch_texts = []
    hotel_sums = np.zeros((n_items, hash_dim), dtype=np.float32)
    flat_view = hotel_sums.reshape(-1)
    n_lines = 0
    n_used = 0

    def flush():
        nonlocal n_used
        if not batch_texts:
            return
        X = vec.transform(batch_texts)
        X_coo = X.tocoo()
        bh = np.asarray(batch_hotels, dtype=np.int64)
        flat_idx = bh[X_coo.row] * hash_dim + X_coo.col
        np.add.at(flat_view, flat_idx, X_coo.data.astype(np.float32))
        n_used += len(batch_texts)
        batch_hotels.clear()
        batch_texts.clear()

    with open(filepath, "rb") as f:
        for line in f:
            n_lines += 1
            obj = _parse(line)
            if obj["date"] > threshold_date:
                continue
            h = item_to_idx.get(obj["hotel_url"], -1)
            if h < 0:
                continue
            text = (obj.get("title") or "") + " " + (obj.get("text") or "")
            batch_hotels.append(h)
            batch_texts.append(text)
            if len(batch_texts) >= BATCH:
                flush()
            if n_lines % 5_000_000 == 0:
                print(f"    {n_lines:,} lines scanned, {n_used:,} used ({time.time()-t0:.0f}s)", flush=True)
    flush()
    print(f"  Counts pass: {n_used:,} reviews ({time.time()-t0:.0f}s); applying IDF + SVD...", flush=True)

    df = (hotel_sums > 0).sum(axis=0).astype(np.float32)
    idf = (np.log((n_items + 1) / (df + 1)) + 1).astype(np.float32)
    hotel_sums *= idf
    norms = np.linalg.norm(hotel_sums, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    hotel_sums /= norms

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    embeddings = svd.fit_transform(hotel_sums).astype(np.float32)
    print(f"  TF-IDF done ({time.time()-t0:.0f}s); SVD explained var: {svd.explained_variance_ratio_.sum():.3f}")
    return embeddings


def _item_aggregates(item_user_mat, global_mean, shrinkage_lambda=10):
    """Per-item shrinkage mean and rating count."""
    csr = item_user_mat
    n_items = csr.shape[0]
    item_means = np.full(n_items, global_mean, dtype=np.float32)
    item_counts = np.zeros(n_items, dtype=np.int32)
    for i in range(n_items):
        s, e = csr.indptr[i], csr.indptr[i + 1]
        n_r = e - s
        item_counts[i] = n_r
        if n_r > 0:
            raw = csr.data[s:e].mean()
            item_means[i] = (n_r * raw + shrinkage_lambda * global_mean) / (n_r + shrinkage_lambda)
    return item_means, item_counts


def _subrating_aggregates(items, item_to_idx, sub_arr, end, shrinkage_lambda=10):
    """Per-item sub-rating mean (shrunk to per-key global mean), variance, coverage."""
    n_items = len(item_to_idx)
    n_keys = sub_arr.shape[1]
    sums = np.zeros((n_items, n_keys), dtype=np.float64)
    sqsums = np.zeros((n_items, n_keys), dtype=np.float64)
    counts = np.zeros((n_items, n_keys), dtype=np.int32)
    total = np.zeros(n_items, dtype=np.int32)

    for j in range(end):
        it = items[j]
        if it not in item_to_idx:
            continue
        idx = item_to_idx[it]
        total[idx] += 1
        row = sub_arr[j]
        for k in range(n_keys):
            v = row[k]
            if not np.isnan(v):
                sums[idx, k] += v
                sqsums[idx, k] += v * v
                counts[idx, k] += 1

    global_counts = counts.sum(axis=0).astype(np.float64)
    global_sums = sums.sum(axis=0)
    global_means = np.where(global_counts > 0, global_sums / np.maximum(global_counts, 1), 0.0)

    means = np.full((n_items, n_keys), np.nan, dtype=np.float32)
    var = np.full((n_items, n_keys), np.nan, dtype=np.float32)
    nz = counts > 0
    counts_f = counts.astype(np.float64)
    raw_means = np.divide(sums, counts_f, where=nz, out=np.zeros_like(sums))
    shrunk = (counts_f * raw_means + shrinkage_lambda * global_means) / (counts_f + shrinkage_lambda)
    means[nz] = shrunk[nz].astype(np.float32)
    nz2 = counts > 1
    var[nz2] = ((sqsums[nz2] / counts[nz2]) - raw_means[nz2] ** 2).astype(np.float32)
    total_safe = np.maximum(total, 1).reshape(-1, 1)
    cov = (counts / total_safe).astype(np.float32)

    # stack as (n_items, 3*n_keys): means | var | coverage
    return np.concatenate([means, var, cov], axis=1)


def _knn_predict(val_u_idx, val_i_idx, val_cold_start_preds, n_items, item_user_mat, knn,
                 user_item_indices, user_item_ratings, user_means, item_temporal,
                 temporal_weight, K, global_mean, desc="Predicting"):
    known_mask = val_i_idx >= 0
    unique_items = np.unique(val_i_idx[known_mask]) if known_mask.any() else np.array([], dtype=np.int32)
    print(f"  Batch KNN for {len(unique_items):,} unique items...", flush=True)
    t3 = time.time()
    if len(unique_items) > 0:
        query_mat = item_user_mat[unique_items]
        distances_all, indices_all = knn.kneighbors(query_mat)
    else:
        distances_all = np.zeros((0, K + 1), dtype=np.float32)
        indices_all = np.zeros((0, K + 1), dtype=np.int32)
    print(f"  done ({time.time()-t3:.0f}s)")

    item_to_query = np.empty(n_items, dtype=np.int32)
    if len(unique_items) > 0:
        item_to_query[unique_items] = np.arange(len(unique_items))

    global _val_u_idx, _val_i_idx, _distances_all, _indices_all
    global _item_to_query, _user_item_indices, _user_item_ratings, _user_means
    global _K, _item_temporal, _progress, _global_mean, _val_cold_start_preds, _temporal_weight
    _val_u_idx = val_u_idx
    _val_i_idx = val_i_idx
    _distances_all = distances_all
    _indices_all = indices_all
    _item_to_query = item_to_query
    _user_item_indices = user_item_indices
    _user_item_ratings = user_item_ratings
    _K = K
    _item_temporal = item_temporal
    _user_means = user_means
    _global_mean = global_mean
    _val_cold_start_preds = val_cold_start_preds
    _temporal_weight = temporal_weight
    _progress = mp.Value("l", 0)

    n = len(val_u_idx)
    chunk_size = (n + N_WORKERS - 1) // N_WORKERS
    pred_args = []
    for w in range(N_WORKERS):
        s = w * chunk_size
        e = min(s + chunk_size, n)
        if s >= e:
            break
        pred_args.append((s, e))

    ctx = mp.get_context("fork")
    with ctx.Pool(N_WORKERS) as pool:
        result_async = pool.map_async(predict_chunk_fork, pred_args)
        with tqdm(total=n, desc=f"{desc} ({N_WORKERS} workers)", unit="sample") as pbar:
            while not result_async.ready():
                result_async.wait(0.1)
                pbar.n = min(_progress.value, n)
                pbar.refresh()
            pbar.n = n
            pbar.refresh()
        pred_results = result_async.get()

    return np.concatenate(pred_results)


def _val_split(users, items, ratings_arr, user_to_idx, item_to_idx, user_means, global_mean,
               start, end):
    """Build (u_idx, i_idx, ratings, cold_start_preds) for rows [start, end)."""
    val_u_idx = np.empty(end - start, dtype=np.int32)
    val_i_idx = np.empty(end - start, dtype=np.int32)
    val_cold_start_preds = np.full(end - start, np.nan, dtype=np.float64)
    cold = 0
    for k, j in enumerate(range(start, end)):
        u = user_to_idx.get(users[j], -1)
        i = item_to_idx.get(items[j], -1)
        val_u_idx[k] = u
        val_i_idx[k] = i
        if u == -1 or i == -1:
            val_cold_start_preds[k] = user_means[u] if u != -1 else global_mean
            cold += 1
    val_ratings = ratings_arr[start:end].astype(np.float32)
    return val_u_idx, val_i_idx, val_ratings, val_cold_start_preds, cold


def run_knn(features, temporal_weight):
    n, dates, users, items, ratings_arr, sub_arr, _geos, t0 = _load_sorted()

    print("Extracting temporal features...", flush=True)
    t1 = time.time()
    temporal_feats, _ = _temporal_features(dates)
    del dates
    print(f"  done ({time.time()-t1:.0f}s), shape: {temporal_feats.shape}")

    train_end = int(n * 0.90)
    val_end = int(n * 0.95)
    print(f"Train: {train_end:,}  Val: {val_end - train_end:,}")

    print("Building indices...", flush=True)
    user_to_idx, item_to_idx = _build_indices(users, items, train_end)
    n_users, n_items = len(user_to_idx), len(item_to_idx)
    print(f"  Users: {n_users:,}  Items: {n_items:,}")

    print("Building item-user matrix...", flush=True)
    item_user_mat = _build_item_user_matrix(users, items, ratings_arr, user_to_idx, item_to_idx, train_end)

    print("Computing per-item temporal features...", flush=True)
    item_temporal = _per_item_temporal(items, item_to_idx, temporal_feats, train_end)

    print(f"Fitting KNN (k={K})...", flush=True)
    t2 = time.time()
    knn = NearestNeighbors(n_neighbors=K + 1, metric="cosine", algorithm="brute", n_jobs=-1)
    knn.fit(item_user_mat)
    print(f"  done ({time.time()-t2:.0f}s)")

    global_mean = float(item_user_mat.data.mean())
    print(f"  Global mean: {global_mean:.3f}")

    print("Building per-user rating lookup...", flush=True)
    user_item_indices, user_item_ratings, user_means, _ = _user_item_lookup(item_user_mat, n_users, global_mean)

    print("Preparing validation set...", flush=True)
    val_u_idx, val_i_idx, val_ratings, val_cold_start_preds, cold = _val_split(
        users, items, ratings_arr, user_to_idx, item_to_idx, user_means, global_mean, train_end, val_end
    )
    print(f"  Total: {len(val_ratings):,}  Cold start: {cold:,}")

    del users, items, ratings_arr, temporal_feats, sub_arr

    tw = temporal_weight if "temporal" in features else 0.0
    t4 = time.time()
    y_pred = _knn_predict(val_u_idx, val_i_idx, val_cold_start_preds, n_items, item_user_mat, knn,
                          user_item_indices, user_item_ratings, user_means, item_temporal,
                          tw, K, global_mean, desc="Predicting")
    print(f"  done ({time.time()-t4:.0f}s)")

    _print_results("KNN CF model", val_ratings, y_pred, global_mean, val_cold_start_preds, cold, t0)


def run_lightgbm(features, temporal_weight, shrinkage_lambda=10, feat_frac=0.80, use_loo=False,
                 tfidf_dims=32, tfidf_vocab_dim=8192,
                 learning_rate=0.05, num_boost_round=500, num_leaves=63, early_stopping=20,
                 residualize=False, use_prior=False,
                 train_frac=0.90, val_frac=0.05):
    try:
        import lightgbm as lgb
    except ImportError:
        sys.exit("LightGBM not installed. Run: pip install lightgbm")

    use_cf = "cf" in features
    use_temporal = "temporal" in features
    use_subratings = "subratings" in features
    use_user_mean = "user_mean" in features
    use_item_mean = "item_mean" in features
    use_user_count = "user_count" in features
    use_item_count = "item_count" in features
    use_location = "location" in features
    use_user_dev = "user_dev" in features
    use_tfidf = "tfidf" in features
    use_user_subratings = "user_subratings" in features
    use_user_var = "user_var" in features
    use_hotel_name = "hotel_name" in features
    use_item_ema = "item_ema" in features
    use_user_ema = "user_ema" in features
    use_date_numeric = "date_numeric" in features

    n, dates, users, items, ratings_arr, sub_arr, geos_arr, t0 = _load_sorted()

    feat_end = int(n * feat_frac)
    lgb_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    overlap = feat_end >= lgb_end
    if overlap:
        feat_end = lgb_end
        lgb_train_start, lgb_train_end = 0, lgb_end
        if use_prior:
            compensation = "time-causal prior encoding"
        elif use_loo:
            compensation = "leave-one-out target encoding"
        else:
            compensation = f"shrinkage λ={shrinkage_lambda}"
        print(f"Feat-train = LGB-train (overlap, leakage handled by {compensation}): {lgb_end:,}  Val: {val_end - lgb_end:,}")
    else:
        lgb_train_start, lgb_train_end = feat_end, lgb_end
        print(f"Feat-train: {feat_end:,}  LGB-train: {lgb_end - feat_end:,}  Val: {val_end - lgb_end:,}  (shrinkage λ={shrinkage_lambda})")

    tfidf_threshold_date = dates[feat_end - 1] if use_tfidf else None

    print("Extracting temporal features...", flush=True)
    t1 = time.time()
    temporal_feats, date_numeric = _temporal_features(dates)
    del dates
    print(f"  done ({time.time()-t1:.0f}s), shape: {temporal_feats.shape}")

    print("Building indices on feat-train slice...", flush=True)
    user_to_idx, item_to_idx = _build_indices(users, items, feat_end)
    n_users, n_items = len(user_to_idx), len(item_to_idx)
    print(f"  Users: {n_users:,}  Items: {n_items:,}")

    print("Building item-user matrix...", flush=True)
    item_user_mat = _build_item_user_matrix(users, items, ratings_arr, user_to_idx, item_to_idx, feat_end)
    global_mean = float(item_user_mat.data.mean())
    print(f"  Global mean: {global_mean:.3f}")

    print("Computing per-user/per-item aggregates...", flush=True)
    user_item_indices, user_item_ratings, user_means, user_counts = _user_item_lookup(
        item_user_mat, n_users, global_mean, shrinkage_lambda=shrinkage_lambda
    )
    item_means, item_counts = _item_aggregates(item_user_mat, global_mean, shrinkage_lambda=shrinkage_lambda)

    item_temporal = None
    if use_cf and use_temporal:
        print("Computing per-item temporal features...", flush=True)
        item_temporal = _per_item_temporal(items, item_to_idx, temporal_feats, feat_end)

    item_subratings = None
    if use_subratings:
        print("Aggregating per-item sub-ratings (mean | var | coverage)...", flush=True)
        item_subratings = _subrating_aggregates(items, item_to_idx, sub_arr, feat_end, shrinkage_lambda=shrinkage_lambda)
        print(f"  shape: {item_subratings.shape}")

    geo_to_idx = None
    region_means = None
    region_counts = None
    if use_location:
        print("Building geo index and per-region aggregates...", flush=True)
        t_g = time.time()
        geo_to_idx, geo_idx_feat = _build_geo_index(geos_arr, feat_end)
        n_regions = len(geo_to_idx)
        region_means, region_counts = _region_aggregates(
            geo_idx_feat, ratings_arr[:feat_end], n_regions, global_mean, shrinkage_lambda=shrinkage_lambda
        )
        del geo_idx_feat
        print(f"  Regions: {n_regions:,}  ({time.time()-t_g:.0f}s)")

    user_devs = None
    user_sr_sums = None
    user_sr_counts = None
    user_sr_global_means = None
    user_sr_means = None
    user_var_sum = None
    user_var_sqsum = None
    user_var_count = None
    user_stds = None
    item_emas = None
    user_emas = None
    item_ema_wsum = None
    item_ema_wrsum = None
    user_ema_wsum = None
    user_ema_wrsum = None
    ema_threshold = None
    if use_user_dev or use_user_subratings or use_user_var or use_item_ema or use_user_ema:
        feat_u_idx, feat_i_idx = _index_arrays(users, items, user_to_idx, item_to_idx, 0, feat_end)
        if use_user_dev:
            print("Computing per-user deviation aggregates...", flush=True)
            t_d = time.time()
            user_devs, _ = _user_dev_aggregates(
                feat_u_idx, feat_i_idx, ratings_arr[:feat_end], item_means, n_users, shrinkage_lambda=shrinkage_lambda
            )
            print(f"  done ({time.time()-t_d:.0f}s)")
        if use_user_subratings:
            print("Computing per-user sub-rating aggregates...", flush=True)
            t_us = time.time()
            user_sr_sums, user_sr_counts, user_sr_global_means = _user_subrating_aggregates(
                feat_u_idx, sub_arr[:feat_end], n_users
            )
            shrunk = ((user_sr_sums + shrinkage_lambda * user_sr_global_means) /
                      (user_sr_counts + shrinkage_lambda)).astype(np.float32)
            user_sr_means = np.where(user_sr_counts > 0, shrunk, np.nan).astype(np.float32)
            print(f"  done ({time.time()-t_us:.0f}s)")
        if use_user_var:
            print("Computing per-user rating variance aggregates...", flush=True)
            t_v = time.time()
            user_var_sum, user_var_sqsum, user_var_count = _user_rating_variance_aggregates(
                feat_u_idx, ratings_arr[:feat_end], n_users
            )
            mean_u = user_var_sum / np.maximum(user_var_count, 1)
            var_u = (user_var_sqsum / np.maximum(user_var_count, 1)) - mean_u * mean_u
            user_stds = np.where(user_var_count > 1,
                                  np.sqrt(np.maximum(var_u, 0)),
                                  np.nan).astype(np.float32)
            print(f"  done ({time.time()-t_v:.0f}s)")
        if use_item_ema or use_user_ema:
            ema_threshold = float(date_numeric[feat_end - 1])
            if use_item_ema:
                print(f"Computing item EMA (taus={EMA_TAUS_DAYS} days)...", flush=True)
                t_ie = time.time()
                item_emas, item_ema_wsum, item_ema_wrsum = _ema_aggregates(
                    feat_i_idx, ratings_arr[:feat_end], date_numeric[:feat_end],
                    ema_threshold, n_items, EMA_TAUS_DAYS,
                )
                print(f"  done ({time.time()-t_ie:.0f}s)")
            else:
                item_ema_wsum = item_ema_wrsum = None
            if use_user_ema:
                print(f"Computing user EMA (taus={EMA_TAUS_DAYS} days)...", flush=True)
                t_ue = time.time()
                user_emas, user_ema_wsum, user_ema_wrsum = _ema_aggregates(
                    feat_u_idx, ratings_arr[:feat_end], date_numeric[:feat_end],
                    ema_threshold, n_users, EMA_TAUS_DAYS,
                )
                print(f"  done ({time.time()-t_ue:.0f}s)")
            else:
                user_ema_wsum = user_ema_wrsum = None
        else:
            ema_threshold = None
            item_ema_wsum = item_ema_wrsum = None
            user_ema_wsum = user_ema_wrsum = None
        del feat_u_idx, feat_i_idx

    hotel_name_emb = None
    if use_hotel_name:
        print("Computing hotel-name TF-IDF embeddings...", flush=True)
        t_n = time.time()
        item_urls = [None] * n_items
        for j in range(feat_end):
            it = items[j]
            idx = item_to_idx.get(it, -1)
            if idx >= 0 and item_urls[idx] is None:
                item_urls[idx] = it
        hotel_name_emb, _vec, expl_var = _hotel_name_embeddings(item_urls)
        print(f"  done ({time.time()-t_n:.0f}s); explained var: {expl_var:.3f}, shape: {hotel_name_emb.shape}")

    hotel_tfidf = None
    if use_tfidf:
        print("Computing hotel TF-IDF embeddings (second file pass)...", flush=True)
        hotel_tfidf = _hotel_tfidf_embeddings(
            FILEPATH, item_to_idx, n_items, tfidf_threshold_date,
            hash_dim=tfidf_vocab_dim, n_components=tfidf_dims,
        )

    knn = None
    if use_cf:
        print(f"Fitting KNN (k={K})...", flush=True)
        t2 = time.time()
        knn = NearestNeighbors(n_neighbors=K + 1, metric="cosine", algorithm="brute", n_jobs=-1)
        knn.fit(item_user_mat)
        print(f"  done ({time.time()-t2:.0f}s)")

    # Build LGB train and val splits (cold-start preds used as cf fallback only)
    lgb_u_idx, lgb_i_idx, lgb_y, lgb_cold_preds, lgb_cold = _val_split(
        users, items, ratings_arr, user_to_idx, item_to_idx, user_means, global_mean, lgb_train_start, lgb_train_end
    )
    val_u_idx, val_i_idx, val_y, val_cold_preds, val_cold = _val_split(
        users, items, ratings_arr, user_to_idx, item_to_idx, user_means, global_mean, lgb_end, val_end
    )
    print(f"LGB-train cold start: {lgb_cold:,}  Val cold start: {val_cold:,}")

    # Compute KNN predictions for both slices in one pass each
    cf_lgb = None
    cf_val = None
    if use_cf:
        tw = temporal_weight if use_temporal else 0.0
        if item_temporal is None:
            # Need a placeholder of correct shape for the worker globals
            item_temporal = np.zeros((n_items, temporal_feats.shape[1]), dtype=np.float32)
        cf_lgb = _knn_predict(lgb_u_idx, lgb_i_idx, lgb_cold_preds, n_items, item_user_mat, knn,
                              user_item_indices, user_item_ratings, user_means, item_temporal,
                              tw, K, global_mean, desc="KNN preds (LGB train)")
        cf_val = _knn_predict(val_u_idx, val_i_idx, val_cold_preds, n_items, item_user_mat, knn,
                              user_item_indices, user_item_ratings, user_means, item_temporal,
                              tw, K, global_mean, desc="KNN preds (val)")

    # Per-row geo indices (if location feature is used)
    geo_idx_lgb = None
    geo_idx_val = None
    if use_location:
        geo_idx_lgb = _geo_idx_for_slice(geos_arr, geo_to_idx, lgb_train_start, lgb_train_end)
        geo_idx_val = _geo_idx_for_slice(geos_arr, geo_to_idx, lgb_end, val_end)

    # Build per-row feature columns: LOO/prior for LGB-train (when enabled), lookup for val
    print("Building LightGBM feature matrices...", flush=True)
    if use_prior:
        if not overlap:
            sys.exit("--prior-encoding requires --feat-frac >= --train-frac (overlap mode); LGB-train must be inside the aggregate slice.")
        print(f"  Time-causal prior encoding on LGB-train...", flush=True)
        t_pr = time.time()
        cols_lgb = _row_features_prior(
            lgb_u_idx, lgb_i_idx, lgb_y, sub_arr[lgb_train_start:lgb_train_end],
            n_users, n_items, global_mean, shrinkage_lambda,
            item_means_for_dev=(item_means if use_user_dev else None),
            geo_idx=geo_idx_lgb, region_means=region_means, region_counts=region_counts,
            hotel_tfidf=hotel_tfidf,
            user_sr_global_means=user_sr_global_means if use_user_subratings else None,
            hotel_name_emb=hotel_name_emb,
            item_emas=item_emas, user_emas=user_emas,
            item_ema_wsum=item_ema_wsum, item_ema_wrsum=item_ema_wrsum,
            user_ema_wsum=user_ema_wsum, user_ema_wrsum=user_ema_wrsum,
            ema_threshold=ema_threshold, ema_taus=EMA_TAUS_DAYS,
            date_numeric_slice=date_numeric[lgb_train_start:lgb_train_end],
        )
        print(f"  prior aggregates done ({time.time()-t_pr:.0f}s)")
    elif use_loo:
        if not overlap:
            sys.exit("--loo requires --feat-frac >= --train-frac (overlap mode); LGB-train must be inside the aggregate slice.")
        print(f"  LOO target encoding on LGB-train...", flush=True)
        t_loo = time.time()
        cols_lgb = _row_features_loo(
            lgb_u_idx, lgb_i_idx, lgb_y, sub_arr[lgb_train_start:lgb_train_end],
            n_users, n_items, global_mean, shrinkage_lambda,
            item_means_for_dev=(item_means if use_user_dev else None),
            geo_idx=geo_idx_lgb, region_means=region_means, region_counts=region_counts,
            hotel_tfidf=hotel_tfidf,
            user_sr_sums=user_sr_sums, user_sr_counts=user_sr_counts,
            user_sr_global_means=user_sr_global_means,
            user_var_sum=user_var_sum, user_var_sqsum=user_var_sqsum, user_var_count=user_var_count,
            hotel_name_emb=hotel_name_emb,
            item_emas=item_emas, user_emas=user_emas,
            item_ema_wsum=item_ema_wsum, item_ema_wrsum=item_ema_wrsum,
            user_ema_wsum=user_ema_wsum, user_ema_wrsum=user_ema_wrsum,
            ema_threshold=ema_threshold, ema_taus=EMA_TAUS_DAYS,
            date_numeric_slice=date_numeric[lgb_train_start:lgb_train_end],
        )
        print(f"  LOO aggregates done ({time.time()-t_loo:.0f}s)")
    else:
        cols_lgb = _row_features_lookup(
            lgb_u_idx, lgb_i_idx, user_means, item_means, user_counts, item_counts, item_subratings,
            user_devs=user_devs, geo_idx=geo_idx_lgb, region_means=region_means, region_counts=region_counts,
            hotel_tfidf=hotel_tfidf, user_sr_means=user_sr_means,
            user_stds=user_stds, hotel_name_emb=hotel_name_emb,
            item_emas=item_emas, user_emas=user_emas,
        )
    cols_val = _row_features_lookup(
        val_u_idx, val_i_idx, user_means, item_means, user_counts, item_counts, item_subratings,
        user_devs=user_devs, geo_idx=geo_idx_val, region_means=region_means, region_counts=region_counts,
        hotel_tfidf=hotel_tfidf, user_sr_means=user_sr_means,
        user_stds=user_stds, hotel_name_emb=hotel_name_emb,
        item_emas=item_emas, user_emas=user_emas,
    )

    X_lgb, feature_names = _build_lgb_features(
        cf_lgb, cols_lgb, temporal_feats[lgb_train_start:lgb_train_end],
        date_numeric[lgb_train_start:lgb_train_end], len(lgb_u_idx),
        use_cf, use_temporal, use_user_mean, use_item_mean, use_user_count, use_item_count,
        use_subratings, use_location, use_user_dev, use_tfidf, use_user_subratings,
        use_user_var, use_hotel_name, use_item_ema, use_user_ema, use_date_numeric,
    )
    X_val, _ = _build_lgb_features(
        cf_val, cols_val, temporal_feats[lgb_end:val_end],
        date_numeric[lgb_end:val_end], len(val_u_idx),
        use_cf, use_temporal, use_user_mean, use_item_mean, use_user_count, use_item_count,
        use_subratings, use_location, use_user_dev, use_tfidf, use_user_subratings,
        use_user_var, use_hotel_name, use_item_ema, use_user_ema, use_date_numeric,
    )
    print(f"  X_lgb: {X_lgb.shape}  X_val: {X_val.shape}")
    print(f"  Features: {feature_names}")

    del users, items, ratings_arr, temporal_feats, sub_arr

    if residualize:
        # Train on rating - item_mean[i]; add baseline back at predict time.
        # For LGB-train: use the LOO/lookup item_mean already in cols_lgb.
        # For val: cols_val item_mean (looked up from full feat-train aggregates).
        # Cold items (NaN item_mean) fall back to global_mean.
        lgb_baseline = np.where(np.isnan(cols_lgb["item_mean"]), global_mean, cols_lgb["item_mean"]).astype(np.float32)
        val_baseline = np.where(np.isnan(cols_val["item_mean"]), global_mean, cols_val["item_mean"]).astype(np.float32)
        lgb_y_train = (lgb_y.astype(np.float32) - lgb_baseline).astype(np.float32)
        val_y_train = (val_y.astype(np.float32) - val_baseline).astype(np.float32)
        print(f"  Residualized target: lgb std {lgb_y.std():.3f} -> {lgb_y_train.std():.3f}, "
              f"val std {val_y.std():.3f} -> {val_y_train.std():.3f}")
    else:
        lgb_baseline = None
        val_baseline = None
        lgb_y_train = lgb_y
        val_y_train = val_y

    print("Training LightGBM...", flush=True)
    t5 = time.time()
    train_set = lgb.Dataset(X_lgb, label=lgb_y_train, feature_name=feature_names)
    val_set = lgb.Dataset(X_val, label=val_y_train, feature_name=feature_names, reference=train_set)
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "verbosity": -1,
    }
    print(f"  LightGBM params: lr={learning_rate}, num_leaves={num_leaves}, "
          f"num_boost_round={num_boost_round}, early_stopping={early_stopping}")
    model = lgb.train(
        params, train_set,
        num_boost_round=num_boost_round,
        valid_sets=[val_set], valid_names=["val"],
        callbacks=[lgb.early_stopping(early_stopping), lgb.log_evaluation(50)],
    )
    print(f"  done ({time.time()-t5:.0f}s, best_iter={model.best_iteration})")

    y_pred = model.predict(X_val, num_iteration=model.best_iteration)
    if residualize:
        y_pred = y_pred + val_baseline
    y_pred = np.clip(y_pred, 1.0, 5.0)

    _print_results("LightGBM model", val_y, y_pred, global_mean, val_cold_preds, val_cold, t0,
                   feature_importance=list(zip(feature_names, model.feature_importance(importance_type="gain"))))


def _row_features_lookup(u_idx, i_idx, user_means, item_means, user_counts, item_counts, item_subratings,
                          user_devs=None, geo_idx=None, region_means=None, region_counts=None,
                          hotel_tfidf=None, user_sr_means=None,
                          user_stds=None, hotel_name_emb=None,
                          item_emas=None, user_emas=None):
    """Per-row feature columns derived from per-user/per-item aggregates (val path, or non-OOF train)."""
    n = len(u_idx)
    valid_u = u_idx >= 0
    valid_i = i_idx >= 0
    safe_u = np.where(valid_u, u_idx, 0)
    safe_i = np.where(valid_i, i_idx, 0)
    cols = {
        "user_mean": np.where(valid_u, user_means[safe_u], np.nan).astype(np.float32),
        "item_mean": np.where(valid_i, item_means[safe_i], np.nan).astype(np.float32),
        "user_count_log": np.where(valid_u, np.log1p(user_counts[safe_u]), np.nan).astype(np.float32),
        "item_count_log": np.where(valid_i, np.log1p(item_counts[safe_i]), np.nan).astype(np.float32),
    }
    if item_subratings is not None:
        sr = np.full((n, item_subratings.shape[1]), np.nan, dtype=np.float32)
        sr[valid_i] = item_subratings[safe_i[valid_i]]
        cols["subratings"] = sr
    if user_devs is not None:
        cols["user_dev"] = np.where(valid_u, user_devs[safe_u], np.nan).astype(np.float32)
    if region_means is not None and geo_idx is not None:
        valid_g = geo_idx >= 0
        safe_g = np.where(valid_g, geo_idx, 0)
        cols["region_mean"] = np.where(valid_g, region_means[safe_g], np.nan).astype(np.float32)
        cols["region_count_log"] = np.where(valid_g, np.log1p(region_counts[safe_g]), np.nan).astype(np.float32)
    if hotel_tfidf is not None:
        emb = np.full((n, hotel_tfidf.shape[1]), np.nan, dtype=np.float32)
        emb[valid_i] = hotel_tfidf[safe_i[valid_i]]
        cols["tfidf"] = emb
    if user_sr_means is not None:
        usr = np.full((n, user_sr_means.shape[1]), np.nan, dtype=np.float32)
        usr[valid_u] = user_sr_means[safe_u[valid_u]]
        cols["user_subratings"] = usr
    if user_stds is not None:
        cols["user_var"] = np.where(valid_u, user_stds[safe_u], np.nan).astype(np.float32)
    if hotel_name_emb is not None:
        emb = np.full((n, hotel_name_emb.shape[1]), np.nan, dtype=np.float32)
        emb[valid_i] = hotel_name_emb[safe_i[valid_i]]
        cols["hotel_name"] = emb
    if item_emas is not None:
        emb = np.full((n, item_emas.shape[1]), np.nan, dtype=np.float32)
        emb[valid_i] = item_emas[safe_i[valid_i]]
        cols["item_ema"] = emb
    if user_emas is not None:
        emb = np.full((n, user_emas.shape[1]), np.nan, dtype=np.float32)
        emb[valid_u] = user_emas[safe_u[valid_u]]
        cols["user_ema"] = emb
    return cols


def _row_features_loo(u_idx, i_idx, ratings, sub_slice, n_users, n_items,
                       global_mean, shrinkage_lambda,
                       item_means_for_dev=None,
                       geo_idx=None, region_means=None, region_counts=None,
                       hotel_tfidf=None,
                       user_sr_sums=None, user_sr_counts=None, user_sr_global_means=None,
                       user_var_sum=None, user_var_sqsum=None, user_var_count=None,
                       hotel_name_emb=None,
                       item_emas=None, user_emas=None,
                       item_ema_wsum=None, item_ema_wrsum=None,
                       user_ema_wsum=None, user_ema_wrsum=None,
                       ema_threshold=None, ema_taus=None, date_numeric_slice=None):
    """Per-row leave-one-out feature columns for LGB-train: subtract each row's own contribution."""
    n = len(u_idx)
    n_keys = sub_slice.shape[1]
    valid_u = u_idx >= 0
    valid_i = i_idx >= 0
    safe_u = np.where(valid_u, u_idx, 0)
    safe_i = np.where(valid_i, i_idx, 0)
    ratings_f64 = ratings.astype(np.float64)

    # Total aggregates over the slice
    total_user_sum = np.zeros(n_users, dtype=np.float64)
    total_user_count = np.zeros(n_users, dtype=np.int64)
    total_item_sum = np.zeros(n_items, dtype=np.float64)
    total_item_count = np.zeros(n_items, dtype=np.int64)
    np.add.at(total_user_sum, u_idx[valid_u], ratings_f64[valid_u])
    np.add.at(total_user_count, u_idx[valid_u], 1)
    np.add.at(total_item_sum, i_idx[valid_i], ratings_f64[valid_i])
    np.add.at(total_item_count, i_idx[valid_i], 1)

    # LOO user/item means: total minus this row's contribution
    user_total_sum = np.where(valid_u, total_user_sum[safe_u], 0.0)
    user_total_count = np.where(valid_u, total_user_count[safe_u], 0)
    item_total_sum = np.where(valid_i, total_item_sum[safe_i], 0.0)
    item_total_count = np.where(valid_i, total_item_count[safe_i], 0)
    loo_user_sum = np.where(valid_u, user_total_sum - ratings_f64, 0.0)
    loo_user_count = np.where(valid_u, user_total_count - 1, 0)
    loo_item_sum = np.where(valid_i, item_total_sum - ratings_f64, 0.0)
    loo_item_count = np.where(valid_i, item_total_count - 1, 0)

    loo_user_mean = np.where(loo_user_count > 0,
                              (loo_user_sum + shrinkage_lambda * global_mean) /
                              (loo_user_count + shrinkage_lambda),
                              np.nan).astype(np.float32)
    loo_item_mean = np.where(loo_item_count > 0,
                              (loo_item_sum + shrinkage_lambda * global_mean) /
                              (loo_item_count + shrinkage_lambda),
                              np.nan).astype(np.float32)
    loo_user_count_log = np.where(loo_user_count > 0, np.log1p(loo_user_count), np.nan).astype(np.float32)
    loo_item_count_log = np.where(loo_item_count > 0, np.log1p(loo_item_count), np.nan).astype(np.float32)

    # Sub-rating totals
    total_sr_sum = np.zeros((n_items, n_keys), dtype=np.float64)
    total_sr_sqsum = np.zeros((n_items, n_keys), dtype=np.float64)
    total_sr_count = np.zeros((n_items, n_keys), dtype=np.int64)
    for k in range(n_keys):
        v = sub_slice[:, k]
        m = ~np.isnan(v) & valid_i
        np.add.at(total_sr_sum[:, k], i_idx[m], v[m].astype(np.float64))
        np.add.at(total_sr_sqsum[:, k], i_idx[m], v[m].astype(np.float64) ** 2)
        np.add.at(total_sr_count[:, k], i_idx[m], 1)
    sr_global_count = total_sr_count.sum(axis=0).astype(np.float64)
    sr_global_sum = total_sr_sum.sum(axis=0)
    sr_global_mean = np.where(sr_global_count > 0, sr_global_sum / np.maximum(sr_global_count, 1), 0.0)

    # LOO sub-rating per key
    loo_sr = np.full((n, n_keys * 3), np.nan, dtype=np.float32)
    for k in range(n_keys):
        v = sub_slice[:, k]
        valid_v = ~np.isnan(v) & valid_i
        v_f64 = np.where(valid_v, v, 0.0).astype(np.float64)
        # Total at this row's item
        item_sr_sum_at_row = np.where(valid_i, total_sr_sum[safe_i, k], 0.0)
        item_sr_sqsum_at_row = np.where(valid_i, total_sr_sqsum[safe_i, k], 0.0)
        item_sr_count_at_row = np.where(valid_i, total_sr_count[safe_i, k], 0)
        # Subtract own contribution if present
        loo_sum = item_sr_sum_at_row - v_f64
        loo_sqsum = item_sr_sqsum_at_row - v_f64 * v_f64
        loo_count = item_sr_count_at_row - valid_v.astype(np.int64)

        denom = loo_count + shrinkage_lambda
        mean_k = np.where(loo_count > 0,
                          (loo_sum + shrinkage_lambda * sr_global_mean[k]) / denom,
                          np.nan).astype(np.float32)
        raw_mean = np.where(loo_count > 0, loo_sum / np.maximum(loo_count, 1), 0.0)
        var_k = np.where(loo_count > 1,
                         (loo_sqsum / np.maximum(loo_count, 1)) - raw_mean * raw_mean,
                         np.nan).astype(np.float32)
        cov_k = np.where(loo_item_count > 0,
                         loo_count / np.maximum(loo_item_count, 1),
                         0.0).astype(np.float32)

        loo_sr[:, k] = mean_k
        loo_sr[:, n_keys + k] = var_k
        loo_sr[:, 2 * n_keys + k] = cov_k

    out = {
        "user_mean": loo_user_mean,
        "item_mean": loo_item_mean,
        "user_count_log": loo_user_count_log,
        "item_count_log": loo_item_count_log,
        "subratings": loo_sr,
    }

    if item_means_for_dev is not None:
        valid_dev = valid_u & valid_i
        row_dev = np.where(valid_dev,
                           ratings_f64 - item_means_for_dev[safe_i].astype(np.float64),
                           0.0)
        total_dev_sum = np.zeros(n_users, dtype=np.float64)
        total_dev_count = np.zeros(n_users, dtype=np.int64)
        np.add.at(total_dev_sum, u_idx[valid_dev], row_dev[valid_dev])
        np.add.at(total_dev_count, u_idx[valid_dev], 1)
        user_dev_sum_at_row = np.where(valid_u, total_dev_sum[safe_u], 0.0)
        user_dev_count_at_row = np.where(valid_u, total_dev_count[safe_u], 0)
        loo_dev_sum = user_dev_sum_at_row - np.where(valid_dev, row_dev, 0.0)
        loo_dev_count = user_dev_count_at_row - valid_dev.astype(np.int64)
        out["user_dev"] = np.where(loo_dev_count > 0,
                                    loo_dev_sum / (loo_dev_count + shrinkage_lambda),
                                    np.nan).astype(np.float32)
    if region_means is not None and geo_idx is not None:
        # Region aggregates aren't OOF'd: region size makes a single row's contribution negligible.
        valid_g = geo_idx >= 0
        safe_g = np.where(valid_g, geo_idx, 0)
        out["region_mean"] = np.where(valid_g, region_means[safe_g], np.nan).astype(np.float32)
        out["region_count_log"] = np.where(valid_g, np.log1p(region_counts[safe_g]), np.nan).astype(np.float32)
    if hotel_tfidf is not None:
        # Hotel-level aggregate: a single review's contribution is small, skip OOF.
        valid_i_o = i_idx >= 0
        safe_i_o = np.where(valid_i_o, i_idx, 0)
        emb = np.full((n, hotel_tfidf.shape[1]), np.nan, dtype=np.float32)
        emb[valid_i_o] = hotel_tfidf[safe_i_o[valid_i_o]]
        out["tfidf"] = emb

    if user_sr_sums is not None:
        # Per-user sub-rating means with LOO: subtract this row's sub-rating contribution per key.
        nk = user_sr_sums.shape[1]
        usr = np.full((n, nk), np.nan, dtype=np.float32)
        for k in range(nk):
            v = sub_slice[:, k]
            valid_v_u = ~np.isnan(v) & valid_u
            v_f64 = np.where(valid_v_u, v, 0.0).astype(np.float64)
            sum_at_row = np.where(valid_u, user_sr_sums[safe_u, k], 0.0)
            count_at_row = np.where(valid_u, user_sr_counts[safe_u, k], 0)
            loo_sum = sum_at_row - v_f64
            loo_count = count_at_row - valid_v_u.astype(np.int64)
            shrunk = ((loo_sum + shrinkage_lambda * user_sr_global_means[k]) /
                      (loo_count + shrinkage_lambda))
            usr[:, k] = np.where(loo_count > 0, shrunk, np.nan).astype(np.float32)
        out["user_subratings"] = usr

    if user_var_sum is not None:
        # LOO per-user rating std: subtract row's contribution.
        sum_at_row = np.where(valid_u, user_var_sum[safe_u], 0.0)
        sqsum_at_row = np.where(valid_u, user_var_sqsum[safe_u], 0.0)
        count_at_row = np.where(valid_u, user_var_count[safe_u], 0)
        loo_sum = np.where(valid_u, sum_at_row - ratings_f64, 0.0)
        loo_sqsum = np.where(valid_u, sqsum_at_row - ratings_f64 * ratings_f64, 0.0)
        loo_count = np.where(valid_u, count_at_row - 1, 0)
        loo_mean = np.where(loo_count > 0, loo_sum / np.maximum(loo_count, 1), 0.0)
        loo_var = np.where(loo_count > 1,
                           (loo_sqsum / np.maximum(loo_count, 1)) - loo_mean * loo_mean,
                           np.nan)
        out["user_var"] = np.where(loo_count > 1, np.sqrt(np.maximum(loo_var, 0)), np.nan).astype(np.float32)

    if hotel_name_emb is not None:
        # Hotel-name TF-IDF is a static per-hotel attribute; no LOO needed.
        valid_i_o = i_idx >= 0
        safe_i_o = np.where(valid_i_o, i_idx, 0)
        emb = np.full((n, hotel_name_emb.shape[1]), np.nan, dtype=np.float32)
        emb[valid_i_o] = hotel_name_emb[safe_i_o[valid_i_o]]
        out["hotel_name"] = emb

    if item_emas is not None:
        # LOO: subtract row's own (weight, weight*rating) from the per-hotel totals before dividing.
        n_taus = item_emas.shape[1]
        emb = np.full((n, n_taus), np.nan, dtype=np.float32)
        days_back = (ema_threshold - date_numeric_slice).astype(np.float64)
        for k, tau in enumerate(ema_taus):
            own_w = np.exp(-days_back / float(tau))
            wsum_at_row = np.where(valid_i, item_ema_wsum[safe_i, k], 0.0)
            wrsum_at_row = np.where(valid_i, item_ema_wrsum[safe_i, k], 0.0)
            loo_w = np.where(valid_i, wsum_at_row - own_w, 0.0)
            loo_wr = np.where(valid_i, wrsum_at_row - own_w * ratings_f64, 0.0)
            emb[:, k] = np.where(loo_w > 1e-9,
                                  loo_wr / np.maximum(loo_w, 1e-12),
                                  np.nan).astype(np.float32)
        out["item_ema"] = emb

    if user_emas is not None:
        n_taus = user_emas.shape[1]
        emb = np.full((n, n_taus), np.nan, dtype=np.float32)
        days_back = (ema_threshold - date_numeric_slice).astype(np.float64)
        for k, tau in enumerate(ema_taus):
            own_w = np.exp(-days_back / float(tau))
            wsum_at_row = np.where(valid_u, user_ema_wsum[safe_u, k], 0.0)
            wrsum_at_row = np.where(valid_u, user_ema_wrsum[safe_u, k], 0.0)
            loo_w = np.where(valid_u, wsum_at_row - own_w, 0.0)
            loo_wr = np.where(valid_u, wrsum_at_row - own_w * ratings_f64, 0.0)
            emb[:, k] = np.where(loo_w > 1e-9,
                                  loo_wr / np.maximum(loo_w, 1e-12),
                                  np.nan).astype(np.float32)
        out["user_ema"] = emb

    return out


def _row_features_prior(u_idx, i_idx, ratings, sub_slice, n_users, n_items,
                         global_mean, shrinkage_lambda,
                         item_means_for_dev=None,
                         geo_idx=None, region_means=None, region_counts=None,
                         hotel_tfidf=None,
                         user_sr_global_means=None,
                         hotel_name_emb=None,
                         item_emas=None, user_emas=None,
                         item_ema_wsum=None, item_ema_wrsum=None,
                         user_ema_wsum=None, user_ema_wrsum=None,
                         ema_threshold=None, ema_taus=None, date_numeric_slice=None):
    """Per-row time-causal aggregates: each row's user/item/sub-rating means use only its group's PRIOR rows."""
    n = len(u_idx)
    n_keys = sub_slice.shape[1]
    valid_u = u_idx >= 0
    valid_i = i_idx >= 0
    safe_u = np.where(valid_u, u_idx, 0)
    safe_i = np.where(valid_i, i_idx, 0)
    ratings_f64 = ratings.astype(np.float64)

    # User/item priors
    pu_sum, pu_cnt = _prior_sum_count(u_idx, ratings_f64)
    pi_sum, pi_cnt = _prior_sum_count(i_idx, ratings_f64)

    prior_user_mean = np.where(pu_cnt > 0,
                                (pu_sum + shrinkage_lambda * global_mean) / (pu_cnt + shrinkage_lambda),
                                np.nan).astype(np.float32)
    prior_item_mean = np.where(pi_cnt > 0,
                                (pi_sum + shrinkage_lambda * global_mean) / (pi_cnt + shrinkage_lambda),
                                np.nan).astype(np.float32)
    prior_user_count_log = np.where(pu_cnt > 0, np.log1p(pu_cnt), np.nan).astype(np.float32)
    prior_item_count_log = np.where(pi_cnt > 0, np.log1p(pi_cnt), np.nan).astype(np.float32)

    # Per-item sub-ratings (mean only; var/cov not produced under prior encoding here)
    prior_sr = np.full((n, n_keys * 3), np.nan, dtype=np.float32)
    for k in range(n_keys):
        v = sub_slice[:, k]
        valid_v = ~np.isnan(v)
        psum, pcnt = _prior_sum_count(i_idx, np.where(valid_v, v, 0.0).astype(np.float64),
                                       valid_value_mask=valid_v)
        # global per-key mean as shrinkage prior (computed from all valid sub-ratings)
        gk = float(np.sum(np.where(valid_v & valid_i, v, 0.0))) / max(float(np.sum(valid_v & valid_i)), 1.0)
        mean_k = np.where(pcnt > 0,
                          (psum + shrinkage_lambda * gk) / (pcnt + shrinkage_lambda),
                          np.nan).astype(np.float32)
        # var: prior mean of squares minus prior mean squared
        psq, pcnt_sq = _prior_sum_count(i_idx, np.where(valid_v, v * v, 0.0).astype(np.float64),
                                         valid_value_mask=valid_v)
        raw_mean = np.where(pcnt > 0, psum / np.maximum(pcnt, 1), 0.0)
        var_k = np.where(pcnt > 1,
                         (psq / np.maximum(pcnt, 1)) - raw_mean * raw_mean,
                         np.nan).astype(np.float32)
        # coverage: pcnt / item_prior_total_count
        cov_k = np.where(pi_cnt > 0,
                         pcnt / np.maximum(pi_cnt, 1),
                         0.0).astype(np.float32)
        prior_sr[:, k] = mean_k
        prior_sr[:, n_keys + k] = var_k
        prior_sr[:, 2 * n_keys + k] = cov_k

    out = {
        "user_mean": prior_user_mean,
        "item_mean": prior_item_mean,
        "user_count_log": prior_user_count_log,
        "item_count_log": prior_item_count_log,
        "subratings": prior_sr,
    }

    # User dev (prior): per-user mean of (rating - item_mean[i_j]) over u's prior reviews
    if item_means_for_dev is not None:
        valid_dev = valid_u & valid_i
        row_dev = np.where(valid_dev,
                           ratings_f64 - item_means_for_dev[safe_i].astype(np.float64),
                           0.0)
        pdev_sum, pdev_cnt = _prior_sum_count(u_idx, row_dev, valid_value_mask=valid_dev)
        out["user_dev"] = np.where(pdev_cnt > 0,
                                    pdev_sum / (pdev_cnt + shrinkage_lambda),
                                    np.nan).astype(np.float32)

    # Per-user sub-rating priors (mean per key)
    if user_sr_global_means is not None:
        usr = np.full((n, n_keys), np.nan, dtype=np.float32)
        for k in range(n_keys):
            v = sub_slice[:, k]
            valid_v = ~np.isnan(v) & valid_u
            psum, pcnt = _prior_sum_count(u_idx, np.where(valid_v, v, 0.0).astype(np.float64),
                                           valid_value_mask=valid_v)
            usr[:, k] = np.where(pcnt > 0,
                                  (psum + shrinkage_lambda * user_sr_global_means[k]) /
                                  (pcnt + shrinkage_lambda),
                                  np.nan).astype(np.float32)
        out["user_subratings"] = usr

    # Per-user rating std (prior)
    pu_sum_v, pu_sq_v, pu_cnt_v = _prior_sum_sqsum_count(u_idx, ratings_f64)
    raw_mean = np.where(pu_cnt_v > 0, pu_sum_v / np.maximum(pu_cnt_v, 1), 0.0)
    var_u = np.where(pu_cnt_v > 1,
                     (pu_sq_v / np.maximum(pu_cnt_v, 1)) - raw_mean * raw_mean,
                     np.nan)
    out["user_var"] = np.where(pu_cnt_v > 1,
                                np.sqrt(np.maximum(var_u, 0.0)),
                                np.nan).astype(np.float32)

    # Region/tfidf/hotel_name/EMA: same as LOO path (per-hotel statics or LOO-decayed)
    if region_means is not None and geo_idx is not None:
        valid_g = geo_idx >= 0
        safe_g = np.where(valid_g, geo_idx, 0)
        out["region_mean"] = np.where(valid_g, region_means[safe_g], np.nan).astype(np.float32)
        out["region_count_log"] = np.where(valid_g, np.log1p(region_counts[safe_g]), np.nan).astype(np.float32)
    if hotel_tfidf is not None:
        emb = np.full((n, hotel_tfidf.shape[1]), np.nan, dtype=np.float32)
        emb[valid_i] = hotel_tfidf[safe_i[valid_i]]
        out["tfidf"] = emb
    if hotel_name_emb is not None:
        emb = np.full((n, hotel_name_emb.shape[1]), np.nan, dtype=np.float32)
        emb[valid_i] = hotel_name_emb[safe_i[valid_i]]
        out["hotel_name"] = emb
    # EMAs: still LOO-style (no per-row prior version implemented)
    if item_emas is not None:
        n_taus = item_emas.shape[1]
        emb = np.full((n, n_taus), np.nan, dtype=np.float32)
        days_back = (ema_threshold - date_numeric_slice).astype(np.float64)
        for k, tau in enumerate(ema_taus):
            own_w = np.exp(-days_back / float(tau))
            wsum_at_row = np.where(valid_i, item_ema_wsum[safe_i, k], 0.0)
            wrsum_at_row = np.where(valid_i, item_ema_wrsum[safe_i, k], 0.0)
            loo_w = np.where(valid_i, wsum_at_row - own_w, 0.0)
            loo_wr = np.where(valid_i, wrsum_at_row - own_w * ratings_f64, 0.0)
            emb[:, k] = np.where(loo_w > 1e-9,
                                  loo_wr / np.maximum(loo_w, 1e-12),
                                  np.nan).astype(np.float32)
        out["item_ema"] = emb
    if user_emas is not None:
        n_taus = user_emas.shape[1]
        emb = np.full((n, n_taus), np.nan, dtype=np.float32)
        days_back = (ema_threshold - date_numeric_slice).astype(np.float64)
        for k, tau in enumerate(ema_taus):
            own_w = np.exp(-days_back / float(tau))
            wsum_at_row = np.where(valid_u, user_ema_wsum[safe_u, k], 0.0)
            wrsum_at_row = np.where(valid_u, user_ema_wrsum[safe_u, k], 0.0)
            loo_w = np.where(valid_u, wsum_at_row - own_w, 0.0)
            loo_wr = np.where(valid_u, wrsum_at_row - own_w * ratings_f64, 0.0)
            emb[:, k] = np.where(loo_w > 1e-9,
                                  loo_wr / np.maximum(loo_w, 1e-12),
                                  np.nan).astype(np.float32)
        out["user_ema"] = emb
    return out


def _build_lgb_features(cf_pred, cols, temporal_rows, date_numeric_rows, n_rows,
                       use_cf, use_temporal, use_user_mean, use_item_mean, use_user_count, use_item_count,
                       use_subratings, use_location, use_user_dev, use_tfidf, use_user_subratings,
                       use_user_var, use_hotel_name, use_item_ema, use_user_ema, use_date_numeric):
    feat = []
    names = []
    if use_cf:
        feat.append(cf_pred.astype(np.float32))
        names.append("cf")
    if use_user_mean:
        feat.append(cols["user_mean"]); names.append("user_mean")
    if use_item_mean:
        feat.append(cols["item_mean"]); names.append("item_mean")
    if use_user_count:
        feat.append(cols["user_count_log"]); names.append("user_count_log")
    if use_item_count:
        feat.append(cols["item_count_log"]); names.append("item_count_log")
    if use_user_dev:
        feat.append(cols["user_dev"]); names.append("user_dev")
    if use_location:
        feat.append(cols["region_mean"]); names.append("region_mean")
        feat.append(cols["region_count_log"]); names.append("region_count_log")
    if use_temporal:
        for k, suffix in enumerate(["doy_sin", "doy_cos"]):
            feat.append(temporal_rows[:, k]); names.append(f"t_{suffix}")
    if use_subratings:
        sr = cols["subratings"]
        n_keys = len(SUBRATING_KEYS)
        # Only emit mean and var; coverage was bottom-of-table importance.
        for tag_idx, tag in enumerate(["mean", "var"]):
            for k, key in enumerate(SUBRATING_KEYS):
                feat.append(sr[:, tag_idx * n_keys + k])
                names.append(f"sub_{key.replace(' ', '_')}_{tag}")
    if use_tfidf:
        emb = cols["tfidf"]
        for k in range(emb.shape[1]):
            feat.append(emb[:, k]); names.append(f"tfidf_{k}")
    if use_user_subratings:
        usr = cols["user_subratings"]
        for k, key in enumerate(SUBRATING_KEYS):
            feat.append(usr[:, k])
            names.append(f"user_sub_{key.replace(' ', '_')}_mean")
    if use_user_var:
        feat.append(cols["user_var"]); names.append("user_rating_std")
    if use_hotel_name:
        emb = cols["hotel_name"]
        for k in range(emb.shape[1]):
            feat.append(emb[:, k]); names.append(f"hname_{k}")
    if use_item_ema:
        emb = cols["item_ema"]
        for k in range(emb.shape[1]):
            feat.append(emb[:, k]); names.append(f"item_ema_{int(EMA_TAUS_DAYS[k])}d")
    if use_user_ema:
        emb = cols["user_ema"]
        for k in range(emb.shape[1]):
            feat.append(emb[:, k]); names.append(f"user_ema_{int(EMA_TAUS_DAYS[k])}d")
    if use_date_numeric:
        feat.append(date_numeric_rows.astype(np.float32))
        names.append("date_numeric")
    X = np.column_stack(feat).astype(np.float32) if feat else np.zeros((n_rows, 0), dtype=np.float32)
    return X, names


def _print_results(model_name, val_ratings, y_pred, global_mean, val_cold_start_preds, cold_start, t0,
                  feature_importance=None):
    y_baseline = np.full(len(val_ratings), global_mean, dtype=np.float32)
    warm_mask = np.isnan(val_cold_start_preds)

    print(f"\n{'='*60}")
    print(f"Results (all {len(val_ratings):,} val samples):")
    print(f"  Cold start: {cold_start:,} ({100*cold_start/len(val_ratings):.1f}%)")
    print(f"  {'':20s} {'MAE':>8s}  {'RMSE':>8s}")
    print(f"  {'Global mean baseline':20s} {mean_absolute_error(val_ratings, y_baseline):8.4f}  {np.sqrt(mean_squared_error(val_ratings, y_baseline)):8.4f}")
    print(f"  {model_name:20s} {mean_absolute_error(val_ratings, y_pred):8.4f}  {np.sqrt(mean_squared_error(val_ratings, y_pred)):8.4f}")

    warm_true = val_ratings[warm_mask]
    warm_pred = y_pred[warm_mask]
    warm_base = y_baseline[warm_mask]
    print(f"\nNon-cold-start only ({warm_mask.sum():,} samples):")
    print(f"  {'':20s} {'MAE':>8s}  {'RMSE':>8s}")
    print(f"  {'Global mean baseline':20s} {mean_absolute_error(warm_true, warm_base):8.4f}  {np.sqrt(mean_squared_error(warm_true, warm_base)):8.4f}")
    print(f"  {model_name:20s} {mean_absolute_error(warm_true, warm_pred):8.4f}  {np.sqrt(mean_squared_error(warm_true, warm_pred)):8.4f}")
    print(f"{'='*60}")

    if feature_importance is not None:
        print("Feature importance (gain):")
        for name, imp in sorted(feature_importance, key=lambda x: -x[1]):
            print(f"  {name:30s} {imp:>12.0f}")

    print(f"Total time: {time.time()-t0:.0f}s")


def _parse_features(arg, method):
    if arg is None:
        return DEFAULT_FEATURES[method]
    feats = [f.strip() for f in arg.split(",") if f.strip()]
    unknown = [f for f in feats if f not in FEATURES]
    if unknown:
        sys.exit(f"Unknown feature(s): {unknown}. Valid: {FEATURES}")
    return feats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=FILEPATH,
                        help=f"Path to the input dataset file (default: {FILEPATH}).")
    parser.add_argument("--baseline", action="store_true", help="Just predict global mean for all samples")
    parser.add_argument("--method", "-m", choices=METHODS, default="knn", help="Model: knn or lightgbm")
    parser.add_argument("--features", "-f", default=None,
                        help=f"Comma-separated subset of {FEATURES}. Default depends on --method.")
    parser.add_argument("--temporal-weight", type=float, default=0.1,
                        help="Temporal blending coefficient for KNN (ignored if 'temporal' not in --features)")
    parser.add_argument("--shrinkage-lambda", type=float, default=10.0,
                        help="Shrinkage strength for user/item/sub-rating means (lightgbm method)")
    parser.add_argument("--feat-frac", type=float, default=0.80,
                        help="Fraction of data used for feature aggregates in lightgbm method. "
                             "If >= train-frac, aggregates and LGB train rows overlap (rely on shrinkage or OOF to compensate).")
    parser.add_argument("--train-frac", type=float, default=0.90,
                        help="Fraction of data used for training (LGB-train end). Default 0.90.")
    parser.add_argument("--val-frac", type=float, default=0.05,
                        help="Fraction of data used for validation (immediately after train). Default 0.05.")
    enc_group = parser.add_mutually_exclusive_group()
    enc_group.add_argument("--loo", action="store_true",
                        help="Leave-one-out target encoding for user_mean/item_mean/sub-ratings/user_dev "
                             "on LGB-train (requires --feat-frac >= 0.90).")
    enc_group.add_argument("--prior-encoding", dest="prior_encoding", action="store_true",
                        help="Time-causal prior encoding: each row's aggregates use only its group's "
                             "rows with strictly earlier dates. Requires --feat-frac >= 0.90.")
    parser.add_argument("--tfidf-dims", type=int, default=32,
                        help="TruncatedSVD output dimensions for the TF-IDF hotel embeddings.")
    parser.add_argument("--tfidf-vocab-dim", type=int, default=8192,
                        help="HashingVectorizer feature space (token hash bucket count).")
    parser.add_argument("--learning-rate", type=float, default=0.05,
                        help="LightGBM learning rate.")
    parser.add_argument("--num-boost-round", type=int, default=500,
                        help="LightGBM max boosting rounds (with early stopping).")
    parser.add_argument("--num-leaves", type=int, default=63,
                        help="LightGBM max leaves per tree.")
    parser.add_argument("--early-stopping", type=int, default=20,
                        help="LightGBM early stopping patience.")
    parser.add_argument("--residualize", action="store_true",
                        help="Train on (rating - item_mean) and add baseline back at predict time.")
    args = parser.parse_args()

    FILEPATH = args.data

    if args.baseline:
        baseline()
    else:
        features = _parse_features(args.features, args.method)
        print(f"Method: {args.method}  Features: {features}")
        if args.method == "knn":
            run_knn(features, args.temporal_weight)
        elif args.method == "lightgbm":
            run_lightgbm(features, args.temporal_weight,
                         shrinkage_lambda=args.shrinkage_lambda,
                         feat_frac=args.feat_frac,
                         use_loo=args.loo,
                         tfidf_dims=args.tfidf_dims,
                         tfidf_vocab_dim=args.tfidf_vocab_dim,
                         learning_rate=args.learning_rate,
                         num_boost_round=args.num_boost_round,
                         num_leaves=args.num_leaves,
                         early_stopping=args.early_stopping,
                         residualize=args.residualize,
                         use_prior=args.prior_encoding,
                         train_frac=args.train_frac,
                         val_frac=args.val_frac)
