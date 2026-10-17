import numpy as np
from audio_utils import pad_to_expected_size
from corner_position import refine_breakpoint, compute_relative_corner_position, calculate_dynamic_rms


def _split_positive_negative_indices(sr, audio):
    positive_indices = []
    negative_indices = []
    for i, val in enumerate(audio):
        if val > 0:
            positive_indices.append([i / sr, val])
        elif val < 0:
            negative_indices.append([i / sr, val])
    return positive_indices, negative_indices


def _detect_breakpoints(audio, rms_pos_arr, rms_neg_arr, peak_height_factor):
    breakpoints = []
    max_start = None
    min_start = None
    max_peak_height_arr = rms_pos_arr * peak_height_factor
    min_peak_height_arr = rms_neg_arr * peak_height_factor
    for i in range(1, len(audio)):
        prev = audio[i - 1]
        curr = audio[i]

        if prev < rms_pos_arr[i - 1] and curr >= rms_pos_arr[i]:
            max_start = i - 1

        if prev > rms_pos_arr[i - 1] and curr <= rms_pos_arr[i]:
            if max_start is None:
                continue
            max_end = i + 1
            max_region = audio[max_start:max_end]
            max_local_idx = np.argmax(max_region)
            max_idx = max_start + max_local_idx
            max_value = audio[max_idx]
            if max_value < max_peak_height_arr[max_idx]:
                continue
            breakpoints.append((max_idx, 1.0))

        if prev > rms_neg_arr[i - 1] and curr <= rms_neg_arr[i]:
            min_start = i - 1

        if prev < rms_neg_arr[i - 1] and curr >= rms_neg_arr[i]:
            if min_start is None:
                continue
            min_end = i + 1
            min_region = audio[min_start:min_end]
            min_local_idx = np.argmin(min_region)
            min_idx = min_start + min_local_idx
            min_value = audio[min_idx]
            if min_value > min_peak_height_arr[min_idx]:
                continue
            breakpoints.append((min_idx, -1.0))

    breakpoints.sort(key=lambda b: b[0])
    return breakpoints


def _refine_breakpoints(audio, breakpoints, window):
    refined_times = []
    refined_signs = []
    indices = [idx for idx, _ in breakpoints]
    for k, (idx, sign) in enumerate(breakpoints):
        gaps = []
        if k > 0:
            gaps.append(idx - indices[k - 1])
        if k < len(indices) - 1:
            gaps.append(indices[k + 1] - idx)
        local_window = max(2, min([window] + [g // 2 for g in gaps]))
        refined_times.append(refine_breakpoint(audio, idx, local_window))
        refined_signs.append(sign)
    return refined_times, refined_signs


def _build_regression_output(audio, breakpoints, corner_positions, corner_duty_cycle, hop_size, window, num_frames):
    refined_times, refined_signs = _refine_breakpoints(audio, breakpoints, window)
    sign_sum = np.zeros(num_frames, dtype=np.float64)
    sign_count = np.zeros(num_frames, dtype=np.float64)
    for t_refined, sign in zip(refined_times, refined_signs):
        frame_idx = int(np.clip(t_refined // hop_size, 0, num_frames - 1))
        sign_sum[frame_idx] += sign
        sign_count[frame_idx] += 1

    has_corner = sign_count > 0
    corner_positions[has_corner] = sign_sum[has_corner] / sign_count[has_corner]

    duty_sum = np.zeros(num_frames, dtype=np.float64)
    duty_weight = np.zeros(num_frames, dtype=np.float64)
    cycles = compute_relative_corner_position(refined_times, refined_signs)
    for t_start, t_end, duty in cycles:
        frame_start = max(int(t_start // hop_size), 0)
        frame_end = min(int(t_end // hop_size), num_frames - 1)
        for f in range(frame_start, frame_end + 1):
            overlap = min(t_end, (f + 1) * hop_size) - max(t_start, f * hop_size)
            if overlap > 0:
                duty_sum[f] += duty * overlap
                duty_weight[f] += overlap

    has_duty = duty_weight > 0
    corner_duty_cycle[has_duty] = duty_sum[has_duty] / duty_weight[has_duty]

    return corner_positions, corner_duty_cycle


def calc_corner_positions(
    audio, hop_size, feat_size, corner_position, sr, contiguous, debug
):
    num_frames = int(np.ceil(len(audio) / hop_size))
    corner_positions = np.zeros(num_frames, dtype=np.float32)
    corner_duty_cycle = np.full(num_frames, -1.0, dtype=np.float32)

    def pad(arr, pad_value):
        return pad_to_expected_size(
            arr, expected_size=feat_size, pad_value=pad_value,
            contiguous=contiguous, debug=debug,
        )

    positive_indices, negative_indices = _split_positive_negative_indices(sr, audio)

    if len(positive_indices) == 0 or len(negative_indices) == 0:
        return pad(corner_positions, 0), pad(corner_duty_cycle, -1.0)
    
    cutoff_hz = corner_position.rms_envelope_cutoff_hz
    rms_pos_arr, rms_neg_arr = calculate_dynamic_rms(audio, sr, cutoff_hz)

    breakpoints = _detect_breakpoints(
        audio, rms_pos_arr, rms_neg_arr, corner_position.peak_height_factor
    )

    corner_positions, corner_duty_cycle = _build_regression_output(
        audio, breakpoints, corner_positions, corner_duty_cycle,
        hop_size, corner_position.regression_window_samples, num_frames,
    )

    return pad(corner_positions, 0), pad(corner_duty_cycle, -1.0)
