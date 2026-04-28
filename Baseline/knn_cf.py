import os
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


def parse_chunk(args):
    filepath, start_byte, end_byte = args
    dates, users, items, ratings = [], [], [], []
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
            items.append(obj["hotel_url"])
            ratings.append(obj["rating"])
    return dates, users, items, ratings


def predict_chunk(args):
    start, end, val_u_idx, val_i_idx, distances_all, indices_all, item_to_query, user_item_indices, user_item_ratings, K = args
    preds = np.full(end - start, np.nan, dtype=np.float64)
    no_pred = 0

    for idx in range(start, end):
        local = idx - start
        u = val_u_idx[idx]
        i = val_i_idx[idx]
        q = item_to_query[i]

        neighbor_dists = distances_all[q]
        neighbor_items = indices_all[q]

        u_items = user_item_indices[u]
        u_ratings = user_item_ratings[u]

        mask = neighbor_items != i
        nb_items = neighbor_items[mask][:K]
        nb_sims = 1.0 - neighbor_dists[mask][:K]

        if len(u_items) == 0:
            no_pred += 1
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
            no_pred += 1

    return preds, no_pred


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
            t_sims = 1.0 - t_dists / (np.sqrt(24) + 1e-9)  # max dist for 6 sin/cos features in [-1,1]
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

    # Flush remaining count
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


def main(temporal_weight=0.1):
    # --- Step 1: Parallel file loading ---
    print(f"Loading data ({N_WORKERS} workers)...", flush=True)
    t0 = time.time()

    file_size = os.path.getsize(FILEPATH)
    boundaries = [i * file_size // N_WORKERS for i in range(N_WORKERS)] + [file_size]
    chunks = [(FILEPATH, boundaries[i], boundaries[i + 1]) for i in range(N_WORKERS)]

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        results = list(pool.map(parse_chunk, chunks))

    # Merge results (order is preserved)
    dates = []
    users = []
    items = []
    ratings = []
    for d, u, it, r in results:
        dates.extend(d)
        users.extend(u)
        items.extend(it)
        ratings.extend(r)
    del results

    n = len(dates)
    print(f"  {n:,} records loaded ({time.time()-t0:.0f}s)")

    # --- Step 2: Sort by date ---
    print("Sorting by date...", flush=True)
    order = np.argsort(dates, kind="mergesort")
    dates = [dates[j] for j in order]
    users = [users[j] for j in order]
    items = [items[j] for j in order]
    ratings_arr = np.array([ratings[j] for j in order], dtype=np.float32)
    del ratings, order

    # --- Step 3: Vectorized temporal feature extraction (sin/cos cyclical encoding) ---
    print("Extracting temporal features...", flush=True)
    t1 = time.time()
    dt = pd.to_datetime(dates)
    dow = dt.dayofweek.values.astype(np.float32)
    doy = dt.dayofyear.values.astype(np.float32)
    mon = dt.month.values.astype(np.float32)
    del dates, dt

    two_pi = 2.0 * np.pi
    # 6 features: sin/cos for each of day-of-week, day-of-year, month
    temporal_feats = np.column_stack([
        np.sin(two_pi * dow / 7.0),
        np.cos(two_pi * dow / 7.0),
        np.sin(two_pi * doy / 365.0),
        np.cos(two_pi * doy / 365.0),
        np.sin(two_pi * mon / 12.0),
        np.cos(two_pi * mon / 12.0),
    ]).astype(np.float32)
    del dow, doy, mon
    print(f"  done ({time.time()-t1:.0f}s), shape: {temporal_feats.shape}")

    # --- Step 4: Split 90% train / 5% val ---
    train_end = int(n * 0.90)
    val_end = int(n * 0.95)
    print(f"Train: {train_end:,}  Val: {val_end - train_end:,}")

    # --- Step 5: Build user/item indices ---
    print("Building indices...", flush=True)
    user_to_idx = {}
    item_to_idx = {}
    for j in range(train_end):
        u, it = users[j], items[j]
        if u not in user_to_idx:
            user_to_idx[u] = len(user_to_idx)
        if it not in item_to_idx:
            item_to_idx[it] = len(item_to_idx)

    n_users = len(user_to_idx)
    n_items = len(item_to_idx)
    print(f"  Users: {n_users:,}  Items: {n_items:,}")

    # --- Step 6: Build item-user sparse matrix (deduplicated) ---
    print("Building item-user matrix...", flush=True)
    row_idx = np.empty(train_end, dtype=np.int32)
    col_idx = np.empty(train_end, dtype=np.int32)
    for j in range(train_end):
        row_idx[j] = item_to_idx[items[j]]
        col_idx[j] = user_to_idx[users[j]]

    pair_rating = {}
    for j in range(train_end):
        pair_rating[(row_idx[j], col_idx[j])] = ratings_arr[j]

    dedup_rows = np.array([k[0] for k in pair_rating], dtype=np.int32)
    dedup_cols = np.array([k[1] for k in pair_rating], dtype=np.int32)
    dedup_data = np.array(list(pair_rating.values()), dtype=np.float32)
    del pair_rating

    print(f"  {train_end:,} -> {len(dedup_data):,} after dedup")
    item_user_mat = csr_matrix(
        (dedup_data, (dedup_rows, dedup_cols)),
        shape=(n_items, n_users), dtype=np.float32,
    )
    del row_idx, col_idx, dedup_rows, dedup_cols, dedup_data

    # --- Step 7: Per-item temporal features (mean of sin/cos encodings) ---
    print("Computing per-item temporal features...", flush=True)
    n_temporal = temporal_feats.shape[1]  # 6
    item_temporal_sum = np.zeros((n_items, n_temporal), dtype=np.float64)
    item_count = np.zeros(n_items, dtype=np.int32)

    for j in range(train_end):
        it = items[j]
        if it in item_to_idx:
            idx = item_to_idx[it]
            item_temporal_sum[idx] += temporal_feats[j]
            item_count[idx] += 1

    item_count_safe = np.maximum(item_count, 1).reshape(-1, 1)
    item_temporal = (item_temporal_sum / item_count_safe).astype(np.float32)
    del item_temporal_sum, item_count, item_count_safe

    # --- Step 8: Fit KNN on sparse CF matrix only (keeps sparsity fast) ---
    # Temporal similarity is blended in during prediction
    print(f"Fitting KNN (k={K})...", flush=True)
    t2 = time.time()
    knn = NearestNeighbors(n_neighbors=K + 1, metric="cosine", algorithm="brute", n_jobs=-1)
    knn.fit(item_user_mat)
    print(f"  done ({time.time()-t2:.0f}s)")

    # --- Step 9: Per-user rating lookup + shrinkage-regularized user means ---
    print("Building per-user rating lookup...", flush=True)
    SHRINKAGE_LAMBDA = 10  # regularization strength
    global_mean = float(item_user_mat.data.mean())
    print(f"  Global mean: {global_mean:.3f}")

    csc = item_user_mat.tocsc()
    user_item_indices = []
    user_item_ratings = []
    user_means = np.full(n_users, global_mean, dtype=np.float32)
    for u in range(n_users):
        start, end = csc.indptr[u], csc.indptr[u + 1]
        user_item_indices.append(csc.indices[start:end].copy())
        user_item_ratings.append(csc.data[start:end].copy())
        n_i = end - start
        if n_i > 0:
            raw_mean = csc.data[start:end].mean()
            user_means[u] = (n_i * raw_mean + SHRINKAGE_LAMBDA * global_mean) / (n_i + SHRINKAGE_LAMBDA)
    del csc

    # --- Step 10: Prepare validation set ---
    # Cold-start users/items get global mean fallback
    print("Preparing validation set...", flush=True)

    val_u_idx = []
    val_i_idx = []
    val_ratings = []
    val_cold_start_preds = []  # pre-filled for cold-start samples
    cold_start = 0
    for j in range(train_end, val_end):
        u, it = users[j], items[j]
        u_idx = user_to_idx.get(u, -1)
        i_idx = item_to_idx.get(it, -1)
        val_u_idx.append(u_idx)
        val_i_idx.append(i_idx)
        val_ratings.append(ratings_arr[j])
        if u_idx == -1 or i_idx == -1:
            # Cold start: use user mean if known, else global mean
            val_cold_start_preds.append(user_means[u_idx] if u_idx != -1 else global_mean)
            cold_start += 1
        else:
            val_cold_start_preds.append(np.nan)

    del users, items, ratings_arr, temporal_feats
    val_u_idx = np.array(val_u_idx, dtype=np.int32)
    val_i_idx = np.array(val_i_idx, dtype=np.int32)
    val_ratings = np.array(val_ratings, dtype=np.float32)
    val_cold_start_preds = np.array(val_cold_start_preds, dtype=np.float64)
    print(f"  Total: {len(val_ratings):,}  Cold start: {cold_start:,}")

    known_mask = val_i_idx >= 0
    unique_items = np.unique(val_i_idx[known_mask])
    print(f"  Batch KNN for {len(unique_items):,} unique items...", flush=True)
    t3 = time.time()
    query_mat = item_user_mat[unique_items]
    distances_all, indices_all = knn.kneighbors(query_mat)
    print(f"  done ({time.time()-t3:.0f}s)")

    item_to_query = np.empty(n_items, dtype=np.int32)
    item_to_query[unique_items] = np.arange(len(unique_items))

    # --- Step 11: Parallel prediction (fork for COW, shared counter + tqdm) ---
    t4 = time.time()

    # Store shared data in module globals so forked children inherit via COW
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

    chunk_size = (len(val_ratings) + N_WORKERS - 1) // N_WORKERS
    pred_args = []
    for w in range(N_WORKERS):
        s = w * chunk_size
        e = min(s + chunk_size, len(val_ratings))
        if s >= e:
            break
        pred_args.append((s, e))

    ctx = mp.get_context("fork")
    total = len(val_ratings)

    with ctx.Pool(N_WORKERS) as pool:
        result_async = pool.map_async(predict_chunk_fork, pred_args)

        with tqdm(total=total, desc=f"Predicting ({N_WORKERS} workers)", unit="sample") as pbar:
            while not result_async.ready():
                result_async.wait(0.1)
                pbar.n = min(_progress.value, total)
                pbar.refresh()
            pbar.n = total
            pbar.refresh()

        pred_results = result_async.get()

    y_pred = np.concatenate(pred_results)
    print(f"  done ({time.time()-t4:.0f}s)")

    # --- Step 12: Results ---
    y_baseline = np.full(len(val_ratings), global_mean, dtype=np.float32)

    non_cold = ~np.isnan(val_cold_start_preds)  # cold start entries are non-NaN
    warm_mask = np.isnan(val_cold_start_preds)   # non-cold-start entries are NaN

    print(f"\n{'='*60}")
    print(f"Results (all {len(val_ratings):,} val samples):")
    print(f"  Cold start: {cold_start:,} ({100*cold_start/len(val_ratings):.1f}%)")
    print(f"  {'':20s} {'MAE':>8s}  {'RMSE':>8s}")
    print(f"  {'Global mean baseline':20s} {mean_absolute_error(val_ratings, y_baseline):8.4f}  {np.sqrt(mean_squared_error(val_ratings, y_baseline)):8.4f}")
    print(f"  {'KNN CF model':20s} {mean_absolute_error(val_ratings, y_pred):8.4f}  {np.sqrt(mean_squared_error(val_ratings, y_pred)):8.4f}")

    warm_true = val_ratings[warm_mask]
    warm_pred = y_pred[warm_mask]
    warm_base = y_baseline[warm_mask]
    print(f"\nNon-cold-start only ({warm_mask.sum():,} samples):")
    print(f"  {'':20s} {'MAE':>8s}  {'RMSE':>8s}")
    print(f"  {'Global mean baseline':20s} {mean_absolute_error(warm_true, warm_base):8.4f}  {np.sqrt(mean_squared_error(warm_true, warm_base)):8.4f}")
    print(f"  {'KNN CF model':20s} {mean_absolute_error(warm_true, warm_pred):8.4f}  {np.sqrt(mean_squared_error(warm_true, warm_pred)):8.4f}")
    print(f"{'='*60}")
    print(f"Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true", help="Just predict global mean for all samples")
    parser.add_argument("--temporal-weight", type=float, default=0.1, help="Temporal blending coefficient (0 = pure CF)")
    args = parser.parse_args()
    if args.baseline:
        baseline()
    else:
        main(args.temporal_weight)
