# !/usr/bin/env python
# -*-coding:utf-8 -*-
import pandas as pd
import random
import numpy as np
from collections import defaultdict


SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def check_label_continuity(df):
    continuity_segments = {}

    for subject in df['subject'].unique():
        subject_data = df[df['subject'] == subject]
        assert subject_data.index[0]==0

        for label in subject_data['activity'].unique():
            label_data = subject_data[subject_data['activity'] == label]

            indices = label_data.index
            segments = []
            start_idx = indices[0]

            for i in range(len(indices) - 1):
                if indices[i] + 1 != indices[i + 1]:
                    end_idx = indices[i]
                    segments.append((start_idx, end_idx))
                    start_idx = indices[i + 1]

            segments.append((start_idx, indices[-1]))

            if segments:
                continuity_segments[(subject, label)] = segments

    return continuity_segments


def split_sequences(sequences, window_size, stride):
    has_null = any(any(pd.isnull(item) or item == '' for item in sublist) for sublist in sequences) # 输出结果
    if has_null:
        raise ValueError("Has null values")

    segments = []
    labels = []

    num_complete_segments = (len(sequences) - window_size) // stride + 1

    for i in range(num_complete_segments):
        start = i * stride
        end = start + window_size
        segment = sequences[start:end]
        assert len(segment) == window_size
        segments.append(np.array(segment))
        labels.append([start, end-1])

    if labels[-1][1] < len(sequences) - 1:
        start = len(sequences) - window_size
        end = len(sequences)
        segment = sequences[start:end]
        assert len(segment) == window_size
        segments.append(np.array(segment))
        labels.append([start, end-1])
    assert len(labels) == len(segments)
    print(f"sequence length: {len(sequences)}\nsegments: {len(segments)}")
    pd.set_option('display.max_columns', None)
    print(labels[:5])
    print(labels[-5:])
    return segments, labels


def split_balanced_by_activity_subject(all_labels, N, seed=SEED):
    """
    all_labels: list of dicts, each with keys "activity" and "subject"
    N: total number of samples to select per activity
    Returns: list of selected indices (length ≈ N * #activities)
    """
    random.seed(seed)

    # Group indices by activity → subject → [indices]
    by_act = defaultdict(lambda: defaultdict(list))
    for i, lbl in enumerate(all_labels):
        by_act[lbl["activity"]][lbl["subject"]].append(i)

    selected_indices = []

    for activity, subj2inds in by_act.items():
        # Flatten into (subject, [indices]) list
        subject_pools = [(subj, inds[:]) for subj, inds in subj2inds.items()]
        random.shuffle(subject_pools)

        # Shuffle within each subject
        for _, inds in subject_pools:
            random.shuffle(inds)

        total_collected = 0
        temp_selected = []

        # 1st pass: try to fairly distribute quota
        S = len(subject_pools)
        base, rem = divmod(N, S)

        # Try base + (1 if rem > 0) per subject
        for i, (subj, inds) in enumerate(subject_pools):
            quota = base + (1 if i < rem else 0)
            taken = inds[:quota]
            temp_selected.extend(taken)
            total_collected += len(taken)
            # update leftover
            subject_pools[i] = (subj, inds[quota:])

        # 2nd pass: top up remaining from any subject with leftovers
        if total_collected < N:
            remaining = N - total_collected
            flat_leftover = [idx for _, inds in subject_pools for idx in inds]
            random.shuffle(flat_leftover)
            temp_selected.extend(flat_leftover[:remaining])

        selected_indices.extend(temp_selected)

    return selected_indices

