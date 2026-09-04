"""NIME scoring — Version 1 (the pipeline used for the reported tables).

    Morlet (band-weighted energy) + first-order derivative on that energy
        → I(t)
    S(t) = I(t) * A(t)
    R(t) = quintile rank of S  (top 20% → rank 1, lowest 20% → rank 5)
    Windows are sorted by mean R (lower = more important); keep top 75%.

Second-order derivatives are not used. W_band lives only inside I(t).
Cross-channel A(t) is a filter, not the definition of R(t).
"""
import os
import pickle
import logging

import numpy as np
import pywt
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_FS = {
    # Experimental pipeline: 256 samples are treated as 1 s (fs = 256 Hz),
    # matching the original main.py and the literature-review convention.
    'DREAMER': 256,
    'STEW': 256,
    'Crowdsource': 256,
    'Crowdsourced': 256,
    'TUAB': 256,
    'TUEV': 256,
    'default': 256,
}

# Task-specific band weights (paper §3.3.2 for TUAB/TUEV; consumer sets as used in experiments).
DATASET_BAND_WEIGHTS = {
    'DREAMER': {'delta': 0.5, 'theta': 0.8, 'alpha': 1.0, 'beta': 1.3, 'gamma': 1.8},
    'Crowdsource': {'delta': 0.6, 'theta': 0.9, 'alpha': 1.5, 'beta': 1.0},
    'Crowdsourced': {'delta': 0.6, 'theta': 0.9, 'alpha': 1.5, 'beta': 1.0},
    'STEW': {'delta': 0.7, 'theta': 1.4, 'alpha': 1.2, 'beta': 1.0},
    'TUAB': {'delta': 1.2, 'theta': 1.2, 'alpha': 1.0, 'beta': 0.8},
    'TUEV': {'delta': 1.4, 'theta': 1.2, 'alpha': 1.2, 'beta': 1.3, 'gamma': 1.3},
}

DATASET_FREQ_BANDS = {
    'DREAMER': {
        'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13),
        'beta': (13, 30), 'gamma': (30, 45),
    },
    'TUEV': {
        'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13),
        'beta': (13, 30), 'gamma': (30, 45),
    },
    'default': {
        'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30),
    },
}


def get_dataset_fs(dataset_name, fs=None):
    if fs is not None:
        return fs
    return DATASET_FS.get(dataset_name, DATASET_FS['default'])


def _minmax(x):
    x = np.asarray(x, dtype=np.float64)
    lo, hi = np.min(x), np.max(x)
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def rank_from_score(S):
    """Quintile ranks of S: top 20% → 1, bottom 20% → 5."""
    S = np.asarray(S, dtype=np.float64)
    R = np.ones(S.shape, dtype=int) * 5
    if np.allclose(S.max(), S.min()):
        return np.ones(S.shape, dtype=int) * 3
    R[S >= np.percentile(S, 20)] = 4
    R[S >= np.percentile(S, 40)] = 3
    R[S >= np.percentile(S, 60)] = 2
    R[S >= np.percentile(S, 80)] = 1
    return R


def _band_weighted_energy(signal_1d, fs, dataset_name):
    """Morlet power, collapsed with task-specific band weights → E(t)."""
    scales = np.arange(1, 128)
    coefficients, frequencies = pywt.cwt(
        signal_1d.astype(np.float64), scales, 'morl', sampling_period=1.0 / fs
    )
    power = np.abs(coefficients) ** 2
    freq_bands = DATASET_FREQ_BANDS.get(dataset_name, DATASET_FREQ_BANDS['default'])
    band_weights = DATASET_BAND_WEIGHTS.get(dataset_name, DATASET_BAND_WEIGHTS['Crowdsource'])
    E = np.zeros(signal_1d.shape[-1], dtype=np.float64)
    band_energy = np.zeros((signal_1d.shape[-1], len(freq_bands)))
    for b_idx, (band_name, (low_f, high_f)) in enumerate(freq_bands.items()):
        freq_mask = (frequencies >= low_f) & (frequencies <= high_f)
        if not np.any(freq_mask):
            continue
        band_power = np.mean(power[freq_mask], axis=0)
        weight = band_weights.get(band_name, 1.0)
        E += band_power * weight
        band_energy[:, b_idx] = band_power * weight
    return E, band_energy


def _energy_first_derivative(E):
    """Forward difference |E(t+1) - E(t)| on Morlet energy (1st derivative only)."""
    D = np.zeros_like(E, dtype=np.float64)
    if E.size > 1:
        D[:-1] = np.abs(E[1:] - E[:-1])
        D[-1] = D[-2]
    return D


def _channel_importance(signal_1d, fs, dataset_name):
    """Cascade: high-energy Morlet region, then 1st derivative of that energy."""
    E, band_energy = _band_weighted_energy(signal_1d, fs, dataset_name)
    D = _energy_first_derivative(E)
    I_c = _minmax(E) * _minmax(D)
    return I_c, E, D, band_energy


def compute_nime_scores(signal, fs=None, dataset_name='default', use_consistency=True):
    """Version-1 point scores.

    Returns dict with I, A, S, R, I_c, band_energy.
    S = I * A  (A ≡ 1 if use_consistency is False).
    R is the quintile rank of S, not of I.
    """
    signal = np.asarray(signal)
    fs = get_dataset_fs(dataset_name, fs)
    if signal.ndim == 1:
        signal = signal[np.newaxis, :]

    C, T = signal.shape
    I_c = np.zeros((C, T), dtype=np.float64)
    band_acc = None
    for ch in range(C):
        I_c[ch], _, _, band_energy = _channel_importance(signal[ch], fs, dataset_name)
        if band_acc is None:
            band_acc = np.zeros_like(band_energy)
        band_acc += band_energy
    band_acc /= max(C, 1)

    I = np.mean(I_c, axis=0)

    if use_consistency and C > 1:
        B = np.zeros((C, T), dtype=bool)
        for ch in range(C):
            tau = np.percentile(I_c[ch], 80)
            B[ch] = I_c[ch] >= tau
        A = np.mean(B, axis=0)
    else:
        A = np.ones(T, dtype=np.float64)

    S = I * A
    R = rank_from_score(S)
    return {
        'I': I,
        'A': A,
        'S': S,
        'ranks': R,
        'I_c': I_c,
        'band_weights': band_acc,
        'channel_agreement': A,
    }


def EEG_Signal_Analysis(signal, fs=None, dataset_name='default', use_consistency=True):
    """Backward-compatible wrapper. combined_feature is I(t)."""
    scores = compute_nime_scores(signal, fs=fs, dataset_name=dataset_name,
                                 use_consistency=use_consistency)
    point_importance = {
        'ranks': scores['ranks'],
        'scores': scores['S'],
        'I': scores['I'],
        'band_weights': scores['band_weights'],
        'channel_agreement': scores['A'],
    }
    return scores['I'], point_importance


def analyze_window_importance(signal, window_size, fs=None, dataset_name='default',
                              threshold_percentage=0.75, use_consistency=True):
    """Keep the top `threshold_percentage` windows ranked by mean R (lower = better)."""
    _, point_importance = EEG_Signal_Analysis(
        signal, fs=fs, dataset_name=dataset_name, use_consistency=use_consistency
    )
    ranks = point_importance['ranks']
    scores = point_importance['scores']
    T = len(ranks)
    window_size = int(window_size)
    if window_size <= 0 or window_size > T:
        return list(range(max(T, 1)))

    window_info = []
    for start in range(T - window_size + 1):
        avg_rank = float(np.mean(ranks[start:start + window_size]))
        avg_score = float(np.mean(scores[start:start + window_size]))
        window_info.append((start, avg_rank, avg_score))

    # Lower rank is more important (Version 1: R comes from S).
    window_info.sort(key=lambda x: x[1])
    n_keep = max(1, int(len(window_info) * threshold_percentage))
    return [start for start, _, _ in window_info[:n_keep]]


def temporal_score_to_patch_scores(S, n_patches):
    """Average S(t) inside each of n_patches equal-length patches (CBraMod)."""
    S = np.asarray(S, dtype=np.float64)
    T = len(S)
    n_patches = int(n_patches)
    patch_len = max(1, T // n_patches)
    scores = np.zeros(n_patches, dtype=np.float64)
    for i in range(n_patches):
        sl = slice(i * patch_len, T if i == n_patches - 1 else (i + 1) * patch_len)
        scores[i] = np.mean(S[sl])
    return scores


def generate_important_timepoints_data(
    data_dict, output_path, target_percentage=0.5, chunk_count=2,
    dataset_name='default', window_threshold=0.75, fs=None,
    use_consistency=True, max_samples=None,
):
    fs = get_dataset_fs(dataset_name, fs)
    important_timepoints = {}
    data_sets = [
        ('train_data', 'train_label'),
        ('val_data', 'val_label'),
        ('test_data', 'test_label'),
        ('All_train_data', 'All_train_label'),
    ]
    sample_counter = 0

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    for data_key, _label_key in data_sets:
        if data_key not in data_dict or data_dict[data_key] is None:
            continue
        data = data_dict[data_key]
        if max_samples is not None and data.shape[0] > max_samples:
            logger.info(f"Limiting {data_key} to {max_samples} samples (from {data.shape[0]})")
            data = data[:max_samples]
        logger.info(f"Processing {data_key} shape={data.shape} dataset={dataset_name} fs={fs}")

        for i in tqdm(range(data.shape[0]), desc=f"Analyzing {data_key}"):
            sample_id = f"{data_key}_{i}"
            signal = data[i]
            total_time_steps = signal.shape[-1]
            chunk_size = max(1, int(total_time_steps * target_percentage) // chunk_count)
            starts = analyze_window_importance(
                signal, chunk_size, fs=fs, dataset_name=dataset_name,
                threshold_percentage=window_threshold, use_consistency=use_consistency,
            )
            important_timepoints[sample_id] = {
                'important_start_points': starts,
                'chunk_size': chunk_size,
                'total_time_steps': total_time_steps,
                'dataset_name': dataset_name,
                'window_threshold': window_threshold,
                'use_consistency': use_consistency,
                'fs': fs,
            }
            sample_counter += 1

    with open(output_path, 'wb') as f:
        pickle.dump(important_timepoints, f)
    logger.info(
        f"Analyzed {sample_counter} samples for {dataset_name} "
        f"(threshold={window_threshold}, consistency={use_consistency}) → {output_path}"
    )
    return important_timepoints


def plot_nime_mask_example(signal, fs, dataset_name, save_path,
                           window_size=None, nime_starts=None, rand_starts=None):
    """Three-row figure: EEG, S(t)/A(t), mask bars (NIME vs random)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    scores = compute_nime_scores(signal, fs=fs, dataset_name=dataset_name)
    T = signal.shape[-1]
    t = np.arange(T) / float(fs)
    gfp = np.mean(np.abs(signal), axis=0)
    if window_size is None:
        window_size = max(1, T // 4)
    if nime_starts is None:
        nime_starts = analyze_window_importance(
            signal, window_size, fs=fs, dataset_name=dataset_name, threshold_percentage=0.75
        )[:2]
    if rand_starts is None:
        rng = np.random.RandomState(0)
        rand_starts = rng.randint(0, max(1, T - window_size + 1), size=2)

    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(t, gfp, color='0.2', lw=0.8)
    axes[0].set_ylabel('EEG |GFP|')
    axes[1].plot(t, scores['S'], label='S(t)=I×A', color='C0')
    axes[1].plot(t, scores['A'], label='A(t)', color='C1', alpha=0.7)
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].set_ylabel('Score')
    ymax = axes[0].get_ylim()[1]
    for s in nime_starts:
        axes[2].axvspan(s / fs, (s + window_size) / fs, color='C0', alpha=0.35, label='NIME')
    for s in rand_starts:
        axes[2].axvspan(s / fs, (s + window_size) / fs, color='C3', alpha=0.25, label='Random')
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel('Mask')
    axes[2].set_xlabel('Time (s)')
    handles, labels = axes[2].get_legend_handles_labels()
    by = dict(zip(labels, handles))
    axes[2].legend(by.values(), by.keys(), loc='upper right', fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path
