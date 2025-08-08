# !/usr/bin/env python
# -*-coding:utf-8 -*-
import numpy as np
from itertools import combinations
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import confusion_matrix
import pandas as pd
import os
import json
from collections import defaultdict
import shutil
from autogluon.tabular import TabularPredictor


def auto_variance_filter(X: pd.DataFrame, feature_cols, primary_thresh=1e-3, second_thresh=1e-2, fallback_thresh=1e-1):
    selector = VarianceThreshold(threshold=primary_thresh)
    selector.fit(X)
    selected_idx = selector.get_support(indices=True)

    if len(selected_idx) > len(feature_cols) / 2:
        selector = VarianceThreshold(threshold=second_thresh)
        selector.fit(X)
        selected_idx = selector.get_support(indices=True)

        if len(selected_idx) > len(feature_cols) / 2:
            selector = VarianceThreshold(threshold=fallback_thresh)
            selector.fit(X)
            selected_idx = selector.get_support(indices=True)

    selected_features = [feature_cols[i] for i in selected_idx]
    print(f"→ Selected {len(selected_features)} / {len(feature_cols)} features after VarianceThreshold")
    return selected_features


def safe_group_split(df, n_splits=4, seed=42, max_tries=100):
    users = np.sort(df['user_id'].unique())
    assert len(users) >= n_splits, "Number of users must ≥ Number of discounts"

    rng = np.random.default_rng(seed)

    for attempt in range(max_tries):
        rng.shuffle(users)
        fold_sizes = np.full(n_splits, len(users) // n_splits)
        fold_sizes[:len(users) % n_splits] += 1

        splits = []
        idx = 0
        for fold_size in fold_sizes:
            val_users = users[idx: idx + fold_size]
            valid_mask = df['user_id'].isin(val_users).to_numpy()
            train_mask = ~valid_mask
            splits.append((
                np.where(train_mask)[0],
                np.where(valid_mask)[0]
            ))
            idx += fold_size
        ok = all(df.iloc[va]['label_bin'].nunique() == 2 for _, va in splits)
        if ok:
            return splits
        rng = np.random.default_rng(seed + attempt + 1)

    raise RuntimeError(f"Attempted {max_tries} times but still cannot guarantee every split of two types of samples")


def pair_wise_feat_importance(all_database_features_df, id2label, json_path, csv_path, model_save_path):
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            nested = json.load(f)
    else:
        nested = {}

    MAX_POS_W = 50
    feature_cols = [c for c in all_database_features_df.columns if c not in ('activity', 'user_id')]
    activities = sorted(all_database_features_df['activity'].unique())

    for A, B in combinations(activities, 2):
        str_a, str_b = str(id2label[A]), str(id2label[B])

        if str_a in nested and str_b in nested[str_a]:
            print(f"Skipping already trained pair: ({str_a} vs {str_b})")
            continue

        # ------------ subset to just the two classes --------------
        sub = all_database_features_df[all_database_features_df['activity'].isin([A, B])].copy()
        sub['label_bin'] = (sub['activity'] == B).astype(int)  # B → 1,  A → 0
        print(f"**************** Training pair **************\n               {id2label[A]} vs {id2label[B]}")
        print("----------" * 4)

        selected_features = auto_variance_filter(sub[feature_cols], feature_cols)

        # Dynamic calculate n_splits based on #users
        users_in_pair = sub["user_id"].unique()
        n_users_in_pair = len(users_in_pair)

        if n_users_in_pair < 2:
            print(f"Skipping pair ({id2label[A]}, {id2label[B]}) — too few users ({n_users_in_pair})")
            continue
        elif n_users_in_pair <= 10:
            N_SPLITS = users_in_pair
        else:
            N_SPLITS = 10

        print(f"→ Using {N_SPLITS} folds for {n_users_in_pair} users")

        importances_accum = np.zeros(len(feature_cols))

        sum_non_zero_vec = 0
        max_time = 10
        tries = 0

        while tries < max_time:

            importances_accum = np.zeros(len(feature_cols))
            n_folds_used = 0

            if tries > 0:
                total_model_path = f"{model_save_path}/{A}_{B}/"
                if os.path.exists(total_model_path):
                    print("Model exists. deleting files...")
                    shutil.rmtree(total_model_path)

            if tries < max_time / 2:
                splits = safe_group_split(sub, n_splits=N_SPLITS, seed=42 + tries)
            else:
                splits = safe_group_split(sub, n_splits=N_SPLITS + 1, seed=42)

            tries += 1

            for fold, (tr_idx, va_idx) in enumerate(splits):
                train_df = sub.iloc[tr_idx].copy().sample(frac=1, random_state=42).reset_index(drop=True)
                valid_df = sub.iloc[va_idx].copy()

                train_users = sorted(train_df['user_id'].unique())
                valid_users = sorted(valid_df['user_id'].unique())

                # calculate pos_w per fold
                n_pos = max((train_df.label_bin == 1).sum(), 1)
                n_neg = (train_df.label_bin == 0).sum()
                pos_w = np.clip(n_neg / n_pos, 1, MAX_POS_W)

                print(f"Fold-{fold}: Train users = {train_users}, Valid users = {valid_users}")
                print(f"Training pair with sample weight: {id2label[A]}: {1.0} vs {id2label[B]}: {pos_w} …")

                train_df['w'] = train_df['label_bin'].map({0: 1.0, 1: pos_w})
                valid_df['w'] = valid_df['label_bin'].map({0: 1.0, 1: pos_w})  # optional

                model_path = f"{model_save_path}/{A}_{B}/fold_{fold}"
                if os.path.exists(model_path):
                    print("Model exists. Loading...")
                    predictor = TabularPredictor.load(model_path)
                else:
                    print("Model not found. Training now...")
                    predictor = TabularPredictor(label='label_bin',
                                                 eval_metric='f1_macro',
                                                 sample_weight='w',
                                                 path=model_path,
                                                 verbosity=0)

                    predictor.fit(
                        train_data=train_df[selected_features + ['label_bin', 'w']],
                        tuning_data=valid_df[selected_features + ['label_bin', 'w']],
                        presets=None,
                        time_limit=10800,  # seconds per fold
                        feature_prune_kwargs=None,  # ← 关掉基于 OOF/permutation 的列剪枝
                        hyperparameters={
                            'GBM': [{}],
                            'CAT': [{}],
                        },
                        num_stack_levels=0
                    )

                # --------- permutation importance on VALID -----------
                fi = predictor.feature_importance(valid_df[selected_features + ['label_bin', 'w']],
                                                  subsample_size=None)
                fi_vec = fi['importance'].reindex(selected_features, fill_value=0.0).values  # align

                fi_vec = np.clip(fi_vec, 0.0, None)
                fi_vec = np.round(fi_vec, 8)

                # Step 2: used feature importance
                fi_dict = dict(zip(selected_features, fi_vec))

                # Step 3: not used features --> 0.0
                fi_vec_full = [fi_dict.get(f, 0.0) for f in feature_cols]

                non_zero_vec = (fi_vec != 0).sum()
                sum_non_zero_vec += non_zero_vec
                print("Non‑zero cols in this fold: ", non_zero_vec,
                      "/", len(fi_vec))

                importances_accum += np.array(fi_vec_full)
                n_folds_used += 1

                y_true = valid_df['label_bin'].values
                y_pred = predictor.predict(valid_df[selected_features])

                n_pos = (y_true == 1).sum()
                n_neg = (y_true == 0).sum()
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

                acc_neg = cm[0, 0] / max(n_neg, 1)  # label 0
                acc_pos = cm[1, 1] / max(n_pos, 1)  # label 1

                print(
                    f"Baseline F1-macro: {predictor.evaluate(valid_df[selected_features + ['label_bin', 'w']], silent=True)['f1_macro']:.4f}")
                print(f"  label 0  | #samples = {n_neg:<4d} | correct = {cm[0, 0]:<4d} | accuracy = {acc_neg:.3f}")
                print(f"  label 1  | #samples = {n_pos:<4d} | correct = {cm[1, 1]:<4d} | accuracy = {acc_pos:.3f}")

                print("----------" * 4)

            if sum_non_zero_vec > 0:
                break
            print("✨All vectors' importance are 0, re-runing...")

        # ------------- fold‑average & normalise -------------------
        w = importances_accum
        if w.sum() > 0:
            w_norm = w / w.sum()
        else:
            w_norm = w  # rare degenerate case

        if str_a not in nested:
            nested[str_a] = {}
        nested[str_a][str_b] = dict(zip(feature_cols, w_norm.tolist()))

        # save to json
        with open(json_path, "w") as f:
            json.dump(nested, f, indent=2)

        # Turn nested into a flat structure：activity_A, activity_B, feature, importance
        rows = []
        for a, b_dict in nested.items():
            for b, fmap in b_dict.items():
                row = {'act_A': a, 'act_B': b}
                # fill every feature, default 0.0
                row.update({feat: fmap.get(feat, 0.0) for feat in feature_cols})
                rows.append(row)

        df_pairs = pd.DataFrame(rows)

        df_pairs = df_pairs[['act_A', 'act_B', *feature_cols]]  # full_feats = list of all feature names
        # save to csv
        df_pairs.to_csv(csv_path, index=False)

        print("----------" * 10)