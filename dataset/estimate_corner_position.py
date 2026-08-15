import librosa
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


def _calculate_split_rms(positive_indices, negative_indices, hop_size):
    if len(positive_indices) == 0 or len(negative_indices) == 0:
        return None, None

    y_pos_vals = [x[1] for x in positive_indices]
    y_neg_vals = [x[1] for x in negative_indices]
    rms_pos = librosa.feature.rms(
        y=np.array(y_pos_vals),
        frame_length=len(positive_indices),
        hop_length=hop_size,
        center=False,
    )
    rms_neg = librosa.feature.rms(
        y=np.array(y_neg_vals),
        frame_length=len(negative_indices),
        hop_length=hop_size,
        center=False,
    )
    return rms_pos[0, 0], -rms_neg[0, 0]


def calc_corner_positions(
    audio, hop_size, feat_size, corner_position, sr, contiguous, debug
):
    use_regression = getattr(corner_position, 'use_regression', False) is True
    use_dynamic_rms = getattr(corner_position, 'use_dynamic_rms', False) is True
    num_frames = int(np.ceil(len(audio) / hop_size))
    corner_positions = np.zeros(num_frames, dtype=np.float32)
    corner_duty_cycle = np.full(num_frames, -1.0, dtype=np.float32) if use_regression else None

    positive_indices, negative_indices = _split_positive_negative_indices(sr, audio)

    if len(positive_indices) == 0 or len(negative_indices) == 0:
        corner_positions = pad_to_expected_size(
            corner_positions,
            expected_size=feat_size,
            pad_value=0,
            contiguous=contiguous,
            debug=debug,
        )
        if use_regression:
            corner_duty_cycle = pad_to_expected_size(
                corner_duty_cycle,
                expected_size=feat_size,
                pad_value=-1.0,
                contiguous=contiguous,
                debug=debug,
            )
        return corner_positions, corner_duty_cycle

    if use_dynamic_rms:
        # Continuously time-varying envelope (paper section 2.2/3.1) instead
        # of one static value over the whole chunk.
        cutoff_hz = corner_position.rms_envelope_cutoff_hz
        rms_pos_arr, rms_neg_arr = calculate_dynamic_rms(audio, sr, cutoff_hz)
    else:
        rms_pos, rms_neg = _calculate_split_rms(
            positive_indices, negative_indices, hop_size
        )
        rms_pos_arr = np.full(len(audio), rms_pos, dtype=np.float32)
        rms_neg_arr = np.full(len(audio), rms_neg, dtype=np.float32)

    # Stage 1 (section 3.1): threshold-crossing peak-pick against the
    # (static or time-varying) RMS+/RMS- envelope - collects (idx, sign)
    # break-points instead of writing directly into corner_positions inline.
    breakpoints = []
    max_start = None
    min_start = None
    min_peak_height_arr = rms_pos_arr * corner_position.peak_height_factor
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
            if max_value < min_peak_height_arr[max_idx]:
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
            if min_value > -min_peak_height_arr[min_idx]:
                continue
            breakpoints.append((min_idx, -1.0))

    breakpoints.sort(key=lambda b: b[0])

    if not use_regression:
        # Stage 1 output only, matching the original behavior exactly.
        for idx, sign in breakpoints:
            corner_positions[idx // hop_size] = sign
    else:
        # Stage 2/3 (sections 3.2, 3.3.2): refine each break-point to
        # sub-sample precision, then derive the relative corner position
        # (duty cycle) per full stick+slip cycle.
        window = corner_position.regression_window_samples
        refined_times = []
        refined_signs = []
        for idx, sign in breakpoints:
            t_refined, _amplitude = refine_breakpoint(audio, idx, window)
            refined_times.append(t_refined)
            refined_signs.append(sign)
            frame_idx = int(np.clip(round(t_refined / hop_size), 0, num_frames - 1))
            corner_positions[frame_idx] = sign

        cycles = compute_relative_corner_position(refined_times, refined_signs)
        for t_start, t_end, duty in cycles:
            frame_start = int(t_start // hop_size)
            frame_end = min(int(t_end // hop_size), num_frames)
            corner_duty_cycle[frame_start:frame_end] = duty

    corner_positions = pad_to_expected_size(
        corner_positions,
        expected_size=feat_size,
        pad_value=0,
        contiguous=contiguous,
        debug=debug,
    )
    if use_regression:
        corner_duty_cycle = pad_to_expected_size(
            corner_duty_cycle,
            expected_size=feat_size,
            pad_value=-1.0,
            contiguous=contiguous,
            debug=debug,
        )
        
    return corner_positions, corner_duty_cycle
