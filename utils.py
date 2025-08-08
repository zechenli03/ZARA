# !/usr/bin/env python
# -*-coding:utf-8 -*-

from typing import List, Dict
from itertools import combinations
import numpy as np
import re, json

def extract_glossary_for_features(
        feature_names: List[str],
        full_glossary: Dict[str, str]
) -> Dict[str, str]:
    selected = {}
    tokens = set()
    for name in feature_names:
        for tk in name.split('_'):
            if tk in full_glossary:
                tokens.add(tk)
    for tk in tokens:
        selected[tk] = full_glossary[tk]

    return selected


def to_row(d, columns):
    cells = [str(d.get(col, "")) for col in columns]
    return "| " + " | ".join(cells) + " |\n"


def sort_by_activity(fused_doc_ids, fused_labels, fused_scores, id2label, top_activity_num):
    from collections import defaultdict

    # 1) collect per‐label lists
    label_dict = defaultdict(list)
    for db_idx, label, sim in zip(fused_doc_ids, fused_labels, fused_scores):
        label_dict[label].append((db_idx, sim))

    # 2) sort each list by sim desc
    for label in label_dict:
        label_dict[label].sort(key=lambda x: x[1], reverse=True)

    # 3) rank labels by their top‐sim score
    ranked = sorted(
        label_dict.keys(),
        key=lambda l: label_dict[l][0][1],
        reverse=True
    )

    # 4) keep only the top N activities
    top_labels = ranked[:top_activity_num]

    # 5) now sort those top labels alphabetically by their name
    top_labels_sorted = sorted(top_labels, key=lambda l: id2label[l])

    # 6) build your outputs in that alphabetical order
    trimmed_label_dict = {l: label_dict[l] for l in top_labels_sorted}
    label_names        = [id2label[l] for l in top_labels_sorted]

    return trimmed_label_dict, label_names


def build_pairwise_prompt_and_list(activity_list, pair_map, top_n=15):
    lines = []
    seen = []
    seen_set = set()

    for a, b in combinations(activity_list, 2):
        fmap = pair_map.get((a, b)) or pair_map.get((b, a), {})
        valid_items = [(f, w) for f, w in fmap.items() if w > 0]
        if not valid_items:
            continue
        top_feats = sorted(valid_items, key=lambda x: x[1], reverse=True)[:top_n]

        entry = f"{a} vs {b}:"
        for feat, w in top_feats:
            entry += f"\n   • {feat} ({round(w*100, 2)}%)"
            if feat not in seen_set:
                seen_set.add(feat)
                seen.append(feat)
        lines.append(entry)

    prompt_str = "\n\n".join(lines)
    unique_feats = sorted(seen)
    return prompt_str, unique_feats


def _smart_round(k: str, v: float) -> float:
    if "freq" in k or "lag" in k:
        return round(v, 3)
    if any(tok in k for tok in ("ratio", "entropy", "skew")):
        return round(v, 4)
    if "tilt" in k:
        return round(v, 2)
    # default for energy-like metrics
    return round(v, 3)


def form_features_mkd_table(feature_list, query_feats, db_feats_df, label_dict, R_NUM,
                            id2label, use_summary=False, label_names=None, N1=4, N2=2):

    query_row = {k: f"{query_feats[k]:.{N1}f}" for k in feature_list}

    table_rows = []
    for lab, refs in label_dict.items():
        label_name = id2label[lab]
        if label_names is not None and label_name not in label_names:
            continue
        mat = []
        for db_idx, _ in refs[:R_NUM]:
            feats_all = db_feats_df.iloc[db_idx].to_dict()
            mat.append([feats_all[k] for k in feature_list])
            if not use_summary:
                table_rows.append(
                    {"label": label_name, **{k: f"{feats_all[k]:.{N1}f}" for k in feature_list}}
                )

        if use_summary and mat:
            arr = np.array(mat)
            means = arr.mean(axis=0)
            stds  = arr.std(axis=0)
            summary = {k: f"{m:.{N1}f}±{s:.{N2}f}"
                       for k, m, s in zip(feature_list, means, stds)}
            table_rows.append({"label": label_name, **summary})

    # compose Markdown table  (header + query row + reference rows)
    columns = ["label"] + feature_list
    header  = "| " + " | ".join(columns) + " |\n"
    sepline = "|---" * len(columns) + "|\n"
    md_lines = [header, sepline]

    # reference rows first
    for row in table_rows:
        md_lines.append(to_row(row, columns))

    # query row
    md_lines.append(to_row({"label": "QUERY", **query_row}, columns))

    return "".join(md_lines)


def is_flat_list_of_scalars(obj):
    return (
        isinstance(obj, list)
        and all(not isinstance(el, (dict, list, tuple, set)) for el in obj)
    )


def safe_fix_json_text(resp):
    """
    Given a raw LLM response (string or object with .text),
    strip Markdown fences and escape newlines/quotes inside "reason" so it's valid JSON.
    """
    # 1) Coerce to plain string
    if not isinstance(resp, (str, bytes)):
        text = getattr(resp, "text", None)
        text = text if isinstance(text, (str, bytes)) else str(resp)
    else:
        text = resp

    # 2) Remove leading/trailing ```json fences
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)

    # 3) Escape newlines and internal quotes in the "reason" value
    #    Pattern: group1="reason":", group2=everything up to the closing quote and comma/brace
    pattern = r'("reason"\s*:\s*")(.+?)(?="(,|\s*\}))'

    def _escape(match):
        body = match.group(2)
        # escape backslashes & quotes first, then newlines
        body = body.replace('\\', '\\\\').replace('"', '\\"')
        body = body.replace('\n', '\\n').replace('\r', '')
        return f'{match.group(1)}{body}'

    return re.sub(pattern, _escape, text, flags=re.DOTALL)


def retrieve_features(md_table: str) -> list[str]:
    features, expected_idx = [], 1
    sep_re = re.compile(r"^-+$")          # match '---'、'------' etc
    for line in md_table.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue

        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 2:
            continue

        if cells[1].lower() == "feature_name" or sep_re.fullmatch(cells[1]):
            continue

        if not cells[0].isdigit() or int(cells[0]) != expected_idx:
            raise ValueError(f"Index error at line: {line}")
        expected_idx += 1

        feature = cells[1]
        if not feature:
            raise ValueError(f"Empty feature name at line: {line}")

        features.append(feature)

    if not features:
        raise ValueError("No feature names extracted—check table format.")

    return features


def retrieve_activities(md_table: str) -> list[str]:
    activities = []
    dash_re = re.compile(r"^-+$")

    for line in md_table.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue

        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 2:
            continue

        if cells[1].lower() == "activity" or dash_re.fullmatch(cells[1]):
            continue

        activities.append(cells[1])

    return activities