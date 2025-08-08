# !/usr/bin/env python
# -*-coding:utf-8 -*-

from typing import Dict
from scipy.signal import stft
from scipy.stats import iqr, skew, kurtosis, entropy
from scipy.signal import welch, detrend, correlate
from scipy.integrate import cumulative_trapezoid
from scipy.signal import butter, filtfilt
from spectrum import arburg
from scipy.spatial.distance import pdist, squareform
import pywt
import numpy as np
import math
import itertools
from collections import Counter

# ────────── constants ──────────
DEG2RAD = math.pi / 180.0
ORDER = 4


def spectral_centroid(freqs, psd):
    return np.sum(freqs * psd) / (np.sum(psd) + 1e-12)


def compute_acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = x - x.mean()
    N = len(x)
    max_lag = min(max_lag, N - 1)

    # All-zero/constant signal processing
    if np.all(x == 0) or np.var(x) < 1e-8:
        acf = np.zeros(max_lag + 1, dtype=float)
        acf[0] = 1.0
        return acf

    r = correlate(x, x, mode='full')
    r = r[N - 1: N - 1 + max_lag + 1]

    denom = r[0] if abs(r[0]) > 1e-12 else 1e-12
    acf = r / denom
    acf[0] = 1.0

    return acf


def zero_cross_centered(channel_data):
    centered = channel_data - np.mean(channel_data)
    signs = np.sign(centered)
    return np.sum(signs[:-1] * signs[1:] < 0)


def band_power(psd_freq, psd_val, fmin, fmax):
    idx = np.logical_and(psd_freq >= fmin, psd_freq <= fmax)
    return np.trapz(psd_val[idx], psd_freq[idx])


def lowpass(data, cutoff, fs, order=3):
    b, a = butter(order, cutoff / (fs / 2), btype='low')
    return filtfilt(b, a, data)


def highpass(data, cutoff, fs, order=3):
    b, a = butter(order, cutoff / (fs / 2), btype='high')
    return filtfilt(b, a, data)


def extract_velocity_features(channel_data, name, axis, fs):
    feats = {}
    sig_hp = highpass(channel_data, cutoff=0.3, fs=fs, order=3)
    vel = cumulative_trapezoid(sig_hp, dx=1 / fs, initial=0)
    vel = detrend(vel, type='linear')
    # avoiad NaN
    vel = np.nan_to_num(vel)

    feats[f"{name}_{axis}_vel_mean"] = float(np.mean(vel))
    feats[f"{name}_{axis}_vel_std"] = float(np.std(vel))
    feats[f"{name}_{axis}_vel_max"] = float(np.max(vel))
    feats[f"{name}_{axis}_vel_min"] = float(np.min(vel))
    feats[f"{name}_{axis}_vel_range"] = float(np.max(vel) - np.min(vel))

    feats[f"{name}_{axis}_vel_rms"] = float(np.sqrt(np.mean(vel ** 2)))
    # slope
    t = np.arange(len(vel))
    num = ((t - t.mean()) * (vel - vel.mean())).sum()
    den = ((t - t.mean()) ** 2).sum()
    feats[f"{name}_{axis}_vel_slope"] = float(num / den if den > 0 else 0.0)
    return feats


def auto_reg_berg(channel_data, name, axis, order=4):
    feats = {}
    ar_coeffs, variance, _ = arburg(channel_data, order)
    n = len(ar_coeffs) - 1

    for k in range(1, n + 1):
        feats[f"{name}_{axis}_ar{k}"] = float(ar_coeffs[k])

    for k in range(n + 1, order + 1):
        feats[f"{name}_{axis}_ar{k}"] = 0.0
    feats[f"{name}_{axis}_ar_var"] = float(variance)
    return feats


def recurrence_rate(channel_data, name, axis, m=2, tau=1, eps=None, exclude_diag=True):
    feats = {}

    L = len(channel_data)
    max_m = (L - 1) // tau + 1
    m_use = min(m, max_m)

    N_embed = L - (m_use - 1) * tau

    X = np.column_stack([channel_data[j * tau: j * tau + N_embed] for j in range(m_use)])
    D = squareform(pdist(X, metric='euclidean'))

    if eps is None:
        eps = 0.1 * np.std(channel_data)
    R = (D <= eps).astype(int)

    if exclude_diag:
        total = N_embed * (N_embed - 1)
        feats[f"{name}_{axis}_rr"] = (R.sum() - N_embed) / total
    else:
        total = N_embed * N_embed
        feats[f"{name}_{axis}_rr"] = R.sum() / total

    return feats


def wavelet_decomposition(channel_data, name, axis, maxlevel=5):
    feats = {}
    wp = pywt.WaveletPacket(data=channel_data,
                            wavelet='db4',
                            mode='symmetric',
                            maxlevel=maxlevel)

    for lvl in range(1, maxlevel + 1):
        nodes = wp.get_level(lvl, order='freq')
        coeffs = np.hstack([n.data for n in nodes])

        sum_abs = np.sum(np.abs(coeffs))
        feats[f"{name}_{axis}_wpd_L{lvl}_sum"] = round(sum_abs, 6)

        energy = np.sum(coeffs ** 2)
        feats[f"{name}_{axis}_wpd_L{lvl}_energy"] = round(energy, 6)

        psq = coeffs ** 2
        p = psq / (np.sum(psq) + 1e-12)
        entropy = -np.sum(p * np.log2(p + 1e-12))
        feats[f"{name}_{axis}_wpd_L{lvl}_entropy"] = round(entropy, 6)

    return feats


def permutation_entropy(channel_data, name, axis, m, tau=1, normalized=False):
    feats = {}

    N = len(channel_data)
    n_windows = N - (m - 1) * tau
    if n_windows <= 0:
        raise ValueError("The sequence is too short to be embedded: please reduce m or tau.")

    perms = list(itertools.permutations(range(m)))
    perm_counts = Counter()

    for i in range(n_windows):
        window = channel_data[i: i + (m - 1) * tau + 1: tau]
        pattern = tuple(np.argsort(window))
        perm_counts[pattern] += 1

    counts = np.array([perm_counts[p] for p in perms], dtype=float)
    probs = counts / counts.sum()
    probs = probs[probs > 0]

    pe = -np.sum(probs * np.log(probs))

    if normalized:
        pe /= math.log(math.factorial(m))

    feats[f"{name}_{axis}_pe"] = float(pe)

    return feats


def extract_time_domain_features(channel_data, name, axis):
    feats: Dict[str, float] = {}
    seq_len = channel_data.shape[0]

    diff = np.diff(channel_data)

    # time-domain
    mean = np.mean(channel_data)
    std = np.std(channel_data)
    maxv = np.max(channel_data)
    minv = np.min(channel_data)
    med = np.median(channel_data)

    rms = np.sqrt(np.mean(channel_data ** 2))
    peak = np.max(np.abs(channel_data))
    var = np.var(channel_data)
    mav = np.mean(np.abs(channel_data))
    sms = np.sum(np.abs(channel_data)) / seq_len

    # slope
    t = np.arange(seq_len, dtype=channel_data.dtype)
    t_mean, channel_data_mean = t.mean(), mean
    num = ((t - t_mean) * (channel_data - channel_data_mean)).sum()
    den = ((t - t_mean) ** 2).sum()
    slope = num / den if den != 0 else 0.0

    zero_crossings = zero_cross_centered(channel_data)
    zc_rate = zero_crossings / (seq_len - 1)

    diff_mean = np.mean(diff) if diff.size > 0 else 0.0
    diff_rms = np.sqrt(np.mean(diff ** 2)) if diff.size > 0 else 0.0
    diff_std = np.std(diff) if diff.size > 0 else 0.0

    prange = maxv - minv
    total = np.sum(channel_data)
    total_abs = np.sum(np.abs(channel_data))

    iqr_v = iqr(channel_data)
    skew_v = skew(channel_data) if std > 0 else 0.0
    kurt_v = kurtosis(channel_data, fisher=False) if std > 0 else 0.0

    feats[f"{name}_{axis}_mean"] = mean
    feats[f"{name}_{axis}_std"] = std
    feats[f"{name}_{axis}_max"] = maxv
    feats[f"{name}_{axis}_min"] = minv
    feats[f"{name}_{axis}_median"] = med

    feats[f"{name}_{axis}_rms"] = rms
    feats[f"{name}_{axis}_peak"] = peak
    feats[f"{name}_{axis}_var"] = var

    feats[f"{name}_{axis}_zc_rate"] = zc_rate

    feats[f"{name}_{axis}_slope"] = slope
    feats[f"{name}_{axis}_diff_mean"] = diff_mean
    feats[f"{name}_{axis}_diff_rms"] = diff_rms
    feats[f"{name}_{axis}_diff_std"] = diff_std

    feats[f"{name}_{axis}_range"] = prange
    feats[f"{name}_{axis}_sum"] = total

    feats[f"{name}_{axis}_sav"] = total_abs
    feats[f"{name}_{axis}_mav"] = mav

    feats[f"{name}_{axis}_iqr"] = iqr_v
    feats[f"{name}_{axis}_skew"] = skew_v
    feats[f"{name}_{axis}_kurtosis"] = kurt_v

    feats[f"{name}_{axis}_sma"] = sms

    return feats


def extract_frequency_domain_features(channel_data, name, axis, fs):
    feat = {}

    # PSD
    signal_len = channel_data.shape[0]
    nperseg = min(256, signal_len)
    nfft = max(256, 2 ** int(np.ceil(np.log2(nperseg))))
    freqs, psd = welch(channel_data, fs=fs, nperseg=nperseg, nfft=nfft, detrend=False)

    # 2. Band power (low/mid/high)
    bands = {"low": (0, 0.5), "mid": (0.5, 3.0), "high": (3.0, min(15.0, freqs[-1]))}
    total_power = np.trapz(psd, freqs) + 1e-8
    for band_name, (low, high) in bands.items():
        if low >= freqs[-1]:
            continue
        bp = band_power(freqs, psd, low, high)
        feat[f"{name}_{axis}_bp_fft_{band_name}"] = bp
        feat[f"{name}_{axis}_bp_fft_{band_name}_ratio"] = bp / total_power

    # 3. Dominant freq & peak
    dom_idx = np.argmax(psd[1:]) + 1  # 跳过 DC
    feat[f"{name}_{axis}_fft_dom_freq"] = freqs[dom_idx]
    feat[f"{name}_{axis}_fft_dom_power"] = psd[dom_idx]

    # 4. FFT （second peak）
    peaks = np.where((psd[1:-1] > psd[:-2]) & (psd[1:-1] > psd[2:]))[0] + 1
    if peaks.size > 1:
        second = peaks[np.argsort(psd[peaks])[-2]]
        feat[f"{name}_{axis}_fft_2nd_peak_freq"] = freqs[second]
        feat[f"{name}_{axis}_fft_2nd_peak_power"] = psd[second]

    # 5. Spectral Centroid
    cent = spectral_centroid(freqs, psd)
    feat[f"{name}_{axis}_fft_sp_centroid"] = cent

    # 6. Spectral Entropy & Flatness
    p_norm = psd / np.sum(psd)
    feat[f"{name}_{axis}_fft_sp_entropy"] = entropy(p_norm)

    # 7. PSD  (skewness)
    feat[f"{name}_{axis}_fft_skew"] = skew(psd)

    # 8. PSD  (kurtosis)
    feat[f"{name}_{axis}_fft_kurtosis"] = kurtosis(psd)

    # 9. weighted average frequency
    ws = np.sum(psd) + 1e-8
    weighted_freq = np.sum(freqs * psd) / ws
    feat[f"{name}_{axis}_fft_weighted_avg_freq"] = weighted_freq

    # 10. spectral energy = ∫PSD² df
    energy = np.trapz(psd ** 2, freqs)
    feat[f"{name}_{axis}_fft_energy"] = energy

    # 11. index of max magnitude in PSD
    max_idx = int(np.argmax(psd))
    feat[f"{name}_{axis}_fft_max_idx"] = max_idx

    return feat


def extract_stft_features(channel_data, name, axis, fs):

    # 1. STFT
    signal_len = channel_data.shape[0]
    # 1. nperseg
    nperseg = min(128, signal_len)
    # 2. noverlap
    noverlap = nperseg // 2
    # 3. nfft
    nfft = 2 ** int(np.ceil(np.log2(nperseg)))

    f, t, Z = stft(channel_data, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft, detrend=False)
    M = np.abs(Z)  # magnitude spectrogram, shape=(len(f), len(t))
    psd = M ** 2

    bands = {"low": (0.0, 0.5), "mid": (0.5, 3.0), "high": (3.0, min(15.0, fs / 2 - 0.1))}
    feats = {}

    for band_name, (low, high) in bands.items():
        mask = (f >= low) & (f < high)
        frame_energy = psd[mask, :].sum(axis=0)  # shape (T,)
        feats[f'{name}_{axis}_stft_{band_name}_max'] = frame_energy.max()
        feats[f'{name}_{axis}_stft_{band_name}_mean'] = frame_energy.mean()
        feats[f'{name}_{axis}_stft_{band_name}_std'] = frame_energy.std()

    p_norm = psd / (psd.sum(axis=0, keepdims=True) + 1e-8)
    ent = -np.sum(p_norm * np.log(p_norm + 1e-12), axis=0)  # per-frame entropy
    feats[f"{name}_{axis}_stft_ent_mean"] = ent.mean()
    feats[f"{name}_{axis}_stft_ent_max"] = ent.max()
    feats[f"{name}_{axis}_stft_ent_std"] = ent.std()

    # per-frame centroid
    cent = np.sum(f[:, None] * psd, axis=0) / (psd.sum(axis=0) + 1e-8)
    feats[f"{name}_{axis}_stft_centroid_mean"] = cent.mean()
    feats[f"{name}_{axis}_stft_centroid_max"] = cent.max()
    feats[f"{name}_{axis}_stft_centroid_std"] = cent.std()

    return feats


def extract_key_acf_features(channel_data, name, axis, max_lag=100) -> dict:
    feats = {}
    acf = compute_acf(channel_data, max_lag)

    # 1) lag>0 and acf[k] > acf[k-1] & acf[k] > acf[k+1]
    first_peak = None
    for k in range(1, max_lag):
        if acf[k] > acf[k - 1] and acf[k] > acf[k + 1]:
            first_peak = k
            break
    feats[f"{name}_{axis}_acf_first_peak_lag"] = first_peak if first_peak is not None else -1.0

    # 2) lag>0 and acf[k] < acf[k-1] & acf[k] < acf[k+1]
    first_min = None
    for k in range(1, max_lag):
        if acf[k] < acf[k - 1] and acf[k] < acf[k + 1]:
            first_min = k
            break
    feats[f"{name}_{axis}_acf_first_min_lag"] = first_min if first_min is not None else -1.0

    # 3) lag>0 and acf[k] <= 0
    first_zero = None
    for k in range(1, max_lag + 1):
        if acf[k] <= 0 and acf[k - 1] > 0:
            denominator = acf[k - 1] - acf[k]
            if abs(denominator) < 1e-12:
                frac = 0.5
            else:
                frac = acf[k - 1] / denominator
            first_zero = (k - 1) + frac
            break
    feats[f"{name}_{axis}_acf_first_zero_lag"] = first_zero if first_zero is not None else -1.0

    return feats


def extract_jerk_features(channel_data, name, axis, fs):
    DT = 1.0 / fs
    jerk = np.diff(channel_data) / DT
    feats = {}

    if np.all(jerk == 0):
        return {
            f"{name}_{axis}_jerk_rms": 0.0,
            f"{name}_{axis}_jerk_peak": 0.0,
            f"{name}_{axis}_jerk_zc_rate": 0.0
        }

    rms = np.sqrt(np.mean(jerk ** 2) + 1e-12)
    peak = np.max(np.abs(jerk))

    signs = np.sign(jerk)
    nonzero_mask = signs != 0
    filtered = signs[nonzero_mask]
    if len(filtered) < 2:
        zc_rate = 0.0
    else:
        zc = np.sum(filtered[:-1] != filtered[1:])
        zc_rate = zc / (len(filtered) - 1)

    feats[f"{name}_{axis}_jerk_rms"] = round(rms, 4)
    feats[f"{name}_{axis}_jerk_peak"] = round(peak, 4)
    feats[f"{name}_{axis}_jerk_zc_rate"] = round(zc_rate, 4)
    return feats


def channel_corr(sensors: dict) -> dict:
    feats = {}

    all_channels = []
    channel_names = []
    for name, mat in sensors.items():
        for i, axis in enumerate(['x', 'y', 'z']):
            all_channels.append(mat[i])
            channel_names.append(f"{name}_{axis}")
    all_data = np.array(all_channels)  # shape: (num_channels, T)

    centered = all_data - all_data.mean(axis=1, keepdims=True)
    corr_mat = np.corrcoef(centered)
    for i in range(len(channel_names)):
        for j in range(i + 1, len(channel_names)):
            key = f"corr_{channel_names[i]}_{channel_names[j]}"
            feats[key] = float(corr_mat[i, j])

    mag_dict = {}
    for name, mat in sensors.items():  # mat: (3, T)
        mag = np.linalg.norm(mat, axis=0)  # shape: (T,)
        mag_dict[f"{name}_mag"] = mag - mag.mean()

    keys = list(mag_dict.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            v1, v2 = mag_dict[keys[i]], mag_dict[keys[j]]
            if v1.std() > 1e-8 and v2.std() > 1e-8:
                corr = np.corrcoef(v1, v2)[0, 1]
            else:
                corr = 0.0
            feats[f"corr_{keys[i]}_{keys[j]}"] = float(corr)

    return feats


def extract_gravity_dynamic_features(acc, prefix, fs):
    feats = {}
    g_x = lowpass(acc[0], cutoff=0.1, fs=fs)
    g_y = lowpass(acc[1], cutoff=0.1, fs=fs)
    g_z = lowpass(acc[2], cutoff=0.1, fs=fs)
    g_mag = np.sqrt(g_x ** 2 + g_y ** 2 + g_z ** 2) + 1e-8
    theta_g = np.arccos(np.clip(g_z / g_mag, -1, 1))
    feats[f"{prefix}_z_grav_angle_mean"] = float(np.mean(theta_g))
    feats[f"{prefix}_z_grav_angle_std"] = float(np.std(theta_g))

    a_dyn_x = highpass(acc[0], cutoff=0.05, fs=fs)
    a_dyn_y = highpass(acc[1], cutoff=0.05, fs=fs)
    a_dyn_z = highpass(acc[2], cutoff=0.05, fs=fs)
    mag_dyn = np.sqrt(a_dyn_x ** 2 + a_dyn_y ** 2 + a_dyn_z ** 2) + 1e-8
    theta_dyn = np.arccos(np.clip(a_dyn_z / mag_dyn, -1, 1))
    feats[f"{prefix}_z_dyn_angle_mean"] = float(np.mean(theta_dyn))
    feats[f"{prefix}_z_dyn_angle_std"] = float(np.std(theta_dyn))
    feats[f"{prefix}_z_dyn_sign"] = float(np.sign(np.mean(a_dyn_z)))

    return feats


def extract_features(sensor_data, fs, channel_names):
    """
    data_: (6, seq_len) numpy.ndarray = [acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z]
    returns: dict of rounded features
    """
    assert type(channel_names) == list
    assert type(sensor_data) == dict

    feats: Dict[str, float] = {}

    for name, sensor in sensor_data.items():
        seq_len = sensor.shape[-1]
        # per-axis features

        # mag = torch.linalg.vector_norm(sensor, dim=0)       # (seq_len,)
        mag = np.linalg.norm(sensor, axis=0)  # (seq_len,)

        for idx, axis in enumerate(("x", "y", "z")):
            channel_data = sensor[idx]  # 1D array, length seq_len

            # time-domain
            feats.update(extract_time_domain_features(channel_data, name, axis))

            feats.update(extract_frequency_domain_features(channel_data, name, axis, fs))
            feats.update(extract_stft_features(channel_data, name, axis, fs))
            feats.update(extract_key_acf_features(channel_data, name, axis, min(fs, seq_len // 2)))
            feats.update(extract_jerk_features(channel_data, name, axis, fs))
            feats.update(auto_reg_berg(channel_data, name, axis, ORDER))
            feats.update(recurrence_rate(channel_data, name, axis, m=2, tau=1, eps=None, exclude_diag=True))
            feats.update(wavelet_decomposition(channel_data, name, axis, maxlevel=5))
            feats.update(permutation_entropy(channel_data, name, axis, m=3, tau=1, normalized=False))

        feats.update(extract_time_domain_features(mag, name, "mag"))
        feats.update(extract_frequency_domain_features(mag, name, "mag", fs))
        feats.update(extract_stft_features(mag, name, "mag", fs))
        feats.update(extract_key_acf_features(mag, name, "mag", min(fs, seq_len // 2)))
        feats.update(extract_jerk_features(mag, name, "mag", fs))
        feats.update(auto_reg_berg(mag, name, "mag", ORDER))
        feats.update(recurrence_rate(mag, name, "mag", m=2, tau=1, eps=None, exclude_diag=True))
        feats.update(wavelet_decomposition(mag, name, "mag", maxlevel=5))
        feats.update(permutation_entropy(mag, name, "mag", m=3, tau=1, normalized=True))
        if "acc" in name:
            feats.update(extract_gravity_dynamic_features(sensor, prefix=name, fs=fs))

    # 3) cross-sensor ratio
    positions = ["T", "RA", "LA", "RL", "LL"]
    for p in positions:
        feats[f"{p}_intensity_ratio_acc_gyro"] = feats[f"{p}_acc_mag_rms"] / (feats[f"{p}_gyro_mag_rms"] + 1e-9)

    # 4)  inter‑channel Pearson correlations
    feats.update(channel_corr(sensor_data))

    return feats
