# !/usr/bin/env python
# -*-coding:utf-8 -*-
import numpy as np
from dtaidistance import dtw_ndim  # pip install fastdtw
import faiss


def z_norm_ts(x: np.ndarray) -> np.ndarray:
    mu = x.mean(0, keepdims=True)
    sig = x.std(0, keepdims=True) + 1e-8
    return (x - mu) / sig


def dtw_dist_multich(ts1: np.ndarray, ts2: np.ndarray) -> float:
    dist = dtw_ndim.distance_fast(ts1.astype(np.float64, copy=False), ts2.astype(np.float64, copy=False),
                                  use_pruning=True)
    return dist


def retrieve_dtw(query_ts: np.ndarray,
                 db_ts: np.ndarray,
                 k: int):
    q = z_norm_ts(query_ts)
    sims = np.empty(len(db_ts), dtype=np.float32)
    for i, cand in enumerate(db_ts):
        sims[i] = -dtw_dist_multich(q, cand)

    top = np.argpartition(-sims, k - 1)[:k]
    top = top[np.argsort(-sims[top])]
    return sims[top], top


def create_index(database_segments):
    database_segments = database_segments.astype(np.float32)
    faiss.normalize_L2(database_segments)

    dim = database_segments.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(database_segments)
    return index


def query_faiss(index, query_embedding, top_k=5000, sim_diff=0.15):
    query_embedding = query_embedding.astype(np.float32)
    faiss.normalize_L2(query_embedding)
    D, I = index.search(query_embedding, top_k)
    D = D.squeeze(0)
    I = I.squeeze(0)

    top1 = D[0]
    threshold = top1 - sim_diff

    mask = D >= threshold

    return D[mask], I[mask]


def rrf_fusion_for_one_query(list_of_indices, K, rrf_k=60):
    candidate_docs = set()
    for indices in list_of_indices:
        candidate_docs.update(indices)

    rank_dicts = []
    for indices in list_of_indices:
        rank_map = {doc_id: rank for rank, doc_id in enumerate(indices)}
        rank_dicts.append(rank_map)

    doc_to_rrf = {}
    for doc_id in candidate_docs:
        score = 0.0
        for rank_map in rank_dicts:
            rank = rank_map.get(doc_id, K)
            score += 1.0 / (rrf_k + rank)
        doc_to_rrf[doc_id] = score

    sorted_doc_ids = sorted(doc_to_rrf, key=doc_to_rrf.get, reverse=True)
    sorted_scores = [doc_to_rrf[doc_id] for doc_id in sorted_doc_ids]

    return sorted_doc_ids, sorted_scores
