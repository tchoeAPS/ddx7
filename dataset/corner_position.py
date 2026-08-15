import numpy as np
from scipy.signal import butter, lfilter


def fit_line(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    return slope, intercept

def refine_breakpoint(audio, idx, window):
    left_start = idx - window
    left_end = idx
    right_start = idx + 1
    right_end = idx + 1 + window

    if left_start < 0 or right_end > len(audio):
        return float(idx), float(audio[idx])

    x_left = np.arange(left_start, left_end)
    y_left = audio[left_start:left_end]
    x_right = np.arange(right_start, right_end)
    y_right = audio[right_start:right_end]

    slope_l, intercept_l = fit_line(x_left, y_left)
    slope_r, intercept_r = fit_line(x_right, y_right)

    if abs(slope_l - slope_r) < 1e-9:
        return float(idx), float(audio[idx])

    t_refined = (intercept_r - intercept_l) / (slope_l - slope_r)
    if t_refined < left_start or t_refined > right_end:
        return float(idx), float(audio[idx])

    return float(t_refined)


def calculate_dynamic_rms(audio, sr, cutoff_hz):
    nyquist = sr / 2.0
    b, a = butter(N=2, Wn=cutoff_hz / nyquist, btype='low')

    pos_energy = np.where(audio > 0, audio ** 2, 0.0)
    neg_energy = np.where(audio < 0, audio ** 2, 0.0)

    filtered_pos = lfilter(b, a, pos_energy)
    filtered_neg = lfilter(b, a, neg_energy)

    rms_pos_env = np.sqrt(np.clip(filtered_pos, 0.0, None))
    rms_neg_env = -np.sqrt(np.clip(filtered_neg, 0.0, None))
    return rms_pos_env, rms_neg_env


def compute_relative_corner_position(breakpoint_times, breakpoint_signs):
    cycles = []
    for k in range(0, len(breakpoint_times) - 2, 2):
        s0, s1, s2 = breakpoint_signs[k], breakpoint_signs[k + 1], breakpoint_signs[k + 2]
        if s0 == s1 or s1 == s2:
            continue
        t0, t1, t2 = breakpoint_times[k], breakpoint_times[k + 1], breakpoint_times[k + 2]
        if not (t0 < t1 < t2):
            continue
        duty = (t1 - t0) / (t2 - t0)
        cycles.append((t0, t2, duty))
    return cycles
