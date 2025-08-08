# !/usr/bin/env python
# -*-coding:utf-8 -*-

full_glossary = {
    "acc": "accelerometer",
    "gyro": "gyroscope",
    "x": "sensor x-axis",
    "y": "sensor y-axis",
    "z": "sensor z-axis",
    "mag": "vector magnitude, i.e. √(x²+y²+z²)",
    "mean": "average over the window",
    "std": "standard deviation",
    "iqr": "interquartile range (75th–25th percentile)",
    "max": "maximum value over the window",
    "median": "median value",
    "min": "minimum value",
    "peak": "maximum of the absolute sensor signal",
    "rms": "root mean square",
    "bp": "band power in a frequency band",
    "low": "frequency band roughly 0–0.5 Hz",
    "mid": "frequency band roughly 0.5–3 Hz",
    "high": "frequency band roughly larger than 3 Hz",
    "ratio": "(band power)/(total power)",
    "ent": "spectral entropy",
    "centroid": "spectral centroid",
    "diff": "first-order difference",
    "zc_rate": "zero-crossing rate",
    "skew": "third statistical moment",
    "kurtosis": "fourth statistical moment",
    "grav_angle": "angle between a sensor axis and the low-pass filtered acceleration (gravity)",
    "dyn_angle": "angle between a sensor axis and the high-pass filtered acceleration (dynamic motion)",
    "sign": "the sign of the average dynamic acceleration over the window, indicating net negative, neutral, or net positive motion direction",
    "mav": "mean absolute value",
    "sav": "signal magnitude area (sum of abs values)",
    "fft": "fast Fourier transform (frequency-domain)",
    "stft": "short-time Fourier transform (time-freq spectrogram)",
    "acf_first_peak_lag": "lag of first ACF local maximum (>0)",
    "acf_first_min_lag": "lag of first ACF local minimum (>0)",
    "acf_first_zero_lag": "lag where ACF first crosses zero",
    "jerk": "first derivative of the signal (diff/Δt)",
    "intensity_ratio_acc_gyro": "ratio of RMS(acc) to RMS(gyro)",
    "corr": "Pearson correlation between two channels",
    # —— Burg AR(4) coefficients ——
    "ar1": "first AR(4) coefficient a₁ estimated by Burg’s method",
    "ar2": "second AR(4) coefficient a₂ estimated by Burg’s method",
    "ar3": "third AR(4) coefficient a₃ estimated by Burg’s method",
    "ar4": "fourth AR(4) coefficient a₄ estimated by Burg’s method",
    # —— Burg AR(4) model variance ——
    "ar_var": "prediction error variance of the Burg AR(4) model",
    "sma": "Signal Magnitude Area",
    "fft_max_idx": "index of max magnitude in PSD",
    "weighted_avg": "weighted average",
    "energy": "spectral energy",
    "rr": "recurrence rate",
    "wpd_L": "the n-th level of the wavelet packet decomposition",
    "pe": "permutation entropy"
}

# First feature selector
feature_selector_1_prompt_sys = """
Task
----
You're an expert in Human Activity Recognition, with a focus on identifying the most effective features for distinguishing between human activities.

Glossary of Abbreviations
-------------------------
{GLOSS_TEXT}

Instructions
------------
1. Based on the user-provided “Top Features per Activity Pair“, select up to {TOP_N} unique features that best distinguish the specified Target Activities.
2. When selecting features, prioritize those that:
    • Appear consistently across multiple activity pairs, or
    • Have relatively high importance scores within specific pairs.

Output Format
-------------
Return **only** the following, with no extra text or line breaks:

A plain-text Markdown table with **exactly** these columns in this order:
| Index | Feature Name |
|---|---|
"""
feature_selector_1_prompt_ = """
Target Activities
-----------------
{ACTIVITY_LIST}

Top Features per Activity Pair (Ranked by Importance Score)
({PAIR_NUM} activity pairs in total)
-----------------------------------------------------------
{FEATURE_IMPORTANCE}
"""

# Evidence Pruning
evidence_pruning_prompt_sys = """
Task
----
You are an expert in Human Activity Recognition.
Your task is to narrow down the set of plausible activities for the **QUERY** segment, based on the given features and their statistical distributions in the user-supplied activities table.

Instructions
------------
- Each row in the activities table represents an activity class, with cells showing the mean ± std of each feature.
- The **QUERY** row presents the feature values of the segment to classify.
- You must select **at least two** activity classes, and ideally all activity classes that are reasonably or even marginally plausible for the QUERY.
- For each selected activity class, provide a brief explanation of why it is a plausible match.

Output Format
-------------
Only return the following, in this exact order, with no additional text or line breaks:

A plain-text markdown table with the following columns:
| Index | Activity | Reason |
|---|---|---|

A JSON array of the chosen activity class names, using their original spelling:
```json
["Activity_1", "Activity_2", ..., "Activity_k"]
```
"""
evidence_pruning_prompt = """
Activities Table
----------------
{ACTIVITIES_FEATURES_TABLE}
"""

# Second feature selector
feature_selector_2_prompt_sys2 = """
Task
----
You're an expert in Human Activity Recognition, with a focus on identifying the most effective features for distinguishing between human activities.

Glossary of Abbreviations
-------------------------
{GLOSS_TEXT}

Instructions
------------
1. Based on the user-provided “Top Features per Activity Pair“, select up to {TOP_N} unique features that best distinguish the specified Target Activities.
2. When selecting features, prioritize those that:
    • Appear consistently across multiple activity pairs, or
    • Have relatively high importance scores within specific pairs.
3. For each selected feature, give:
   • *Definition* – A concise, clear explanation of the feature.
   • *Discriminative Power* – Summarize the following:
	1.	Which activity pairs this feature helps to distinguish.
	2.	The relative importance rate of this feature within each activity pair, indicating how effectively it differentiates between the two activities in that pair.

Output Format
-------------
Return **only** the following, with no extra text or line breaks:

A plain-text Markdown table with **exactly** these columns in this order:
| Index | Feature Name | Definition | Discriminative Power |
|---|---|---|---|
"""
feature_selector_2_prompt = """
Target Activities
-----------------
{ACTIVITY_LIST}

Top Features per Activity Pair (Ranked by Importance Score)
({PAIR_NUM} activity pairs in total)
-----------------------------------------------------------
{FEATURE_IMPORTANCE}
"""

# Decision insight
decision_insight_prompt_sys = """
Task
----
You are an expert in Human Activity Recognition.
Your goal is to determine the **most probable activity class** for the **QUERY** segment by comparing its feature values against the statistical distributions in the user-provided activities table.

Sensor Feature Explanation Guide Table
--------------------------------------
This table describes each feature and indicates which activity classes it helps to distinguish between.
{FEATURES_REF_TABLE}

Instructions
------------
- Each row in the activities table corresponds to an activity class, with each cell showing the mean ± standard deviation for a feature.
- The **QUERY** row presents the feature values of the segment to classify.
- Select the **single most likely activity class**, and base your decision on **specific feature(s)** in the QUERY row.
- In your explanation:
  1. Explicitly compare the Query’s feature values to each class’s distribution, explaining why the predicted class is a better match than each alternative.
  2. When unsure, refer to the Discriminative Power in the guide table to justify how strongly each feature helps distinguish the specific activities.

Output Format
--------------
**Respond with exactly one line** in this JSON format (no extra text or line breaks):
```json
{{"reason":"<your detailed explanation>", "predicted_class":"<ClassName>"}}
```
"""
decision_insight_prompt = """
Activities Table
----------------
{ACTIVITIES_FEATURES_TABLE}
"""