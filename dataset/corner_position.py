import numpy as np
from scipy.signal import butter, lfilter

'''
Shared corner-position primitives used by both the production dataset
pipeline (create_data.py) and the debug/visualization script
(estimate_corner_position.py).

These implement Stage 2 (regression-based break-point refinement) and
Stage 3 (relative corner position / duty cycle) from Buys & McPherson,
"Real-Time Bowed String Feature Extraction for Performance Applications"
(SMC 2018), using a simple closed-form two-segment least-squares fit per
corner rather than the paper's full iterative Muggeo procedure.
'''

def fit_line(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    return slope, intercept

def refine_breakpoint(audio, idx, window):
    '''
    Refine a Stage-1 (threshold-crossing peak-pick) break-point at sample
    index `idx` to sub-sample precision.

    Fits one line to the `window` samples strictly before `idx` and another
    to the `window` samples strictly after `idx` (skipping the sample at
    `idx` itself, since corners are rounded in practice - see paper section
    2.1), then solves for the intersection of the two lines.

    Falls back to `(idx, audio[idx])` unrefined if there aren't enough
    samples on either side to fit a line (e.g. near a segment boundary), or
    if the fitted lines are near-parallel / their intersection lands outside
    the fitted window - both signs that the local data doesn't actually look
    like two clean line segments (e.g. noise, a multiple-flyback/multiple-
    slip region - see paper section 2.1) and extrapolating would be
    unreliable rather than merely imprecise.
    '''
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

    amplitude = slope_l * t_refined + intercept_l
    return float(t_refined), float(amplitude)


def calculate_dynamic_rms(audio, sr, cutoff_hz):
    '''
    Continuously time-varying RMS+/RMS- envelope, per the paper's actual
    section 2.2/3.1 methodology: rectify the signal into positive/negative
    energy, then low-pass filter each to get a smoothly-varying amplitude
    envelope (rather than one static scalar over the whole chunk).

    Uses a 2nd-order causal Butterworth low-pass filter (consistent with the
    paper's real-time framing) at `cutoff_hz`. The paper reports 150 Hz as
    empirically suitable across all four violin strings: low enough to be
    independent of the oscillation cycles themselves, high enough to still
    follow transient amplitude changes.

    Returns (rms_pos_env, rms_neg_env), two arrays the same length as
    `audio`; rms_neg_env is kept negative, mirroring calculate_split_rms.
    '''
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
    '''
    Given a time-ordered sequence of refined break-point times and their
    signs (+1.0 for a max/stick-to-slip corner, -1.0 for a min), compute the
    "relative corner position" (duty cycle) for each full oscillation cycle.

    A full cycle spans three consecutive break-points (t_k, t_k+1, t_k+2),
    i.e. one stick segment followed by one slip segment, which requires the
    signs to alternate (max, min, max or min, max, min). Cycles do not
    overlap: each one consumes 2 break-points (one full period), and the
    next cycle starts where it left off - t_k+2 becomes the next cycle's
    t_k - rather than sliding by 1, which would alternate between measuring
    stick-then-slip and slip-then-stick and report `duty` and `1 - duty` on
    alternating windows for the same physical signal.

    Non-alternating triples (e.g. from a "multiple flyback"/"multiple slip"
    detection glitch, see paper section 2.1) don't represent a real cycle and
    are skipped. Each break-point is refined independently from its own
    local window (see refine_breakpoint), so on closely-spaced or noisy
    corners a refined time can end up out of order relative to its
    neighbors; triples that aren't strictly increasing (t0 < t1 < t2) are
    skipped too, since duty would otherwise fall outside [0, 1].

    Returns a list of (t_start, t_end, duty) tuples, one per detected cycle,
    where t_start/t_end are the cycle's bounding break-point times (in
    samples) and duty is in [0, 1].
    '''
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
