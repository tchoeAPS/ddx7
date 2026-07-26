import librosa
import numpy as np
from audio_utils import pad_to_expected_size


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
    if debug:
        print(f"[DEBUG] calc_corner_positions audio shape: {audio.shape}")
        print(f"[DEBUG] audio min/max: {audio.min()} / {audio.max()}")

    corner_positions = np.zeros(
        int(np.ceil(len(audio) / hop_size)),
        dtype=np.float32,
    )
    positive_indices, negative_indices = _split_positive_negative_indices(sr, audio)
    if debug:
        print(f"[DEBUG] positive samples: {len(positive_indices)}")
        print(f"[DEBUG] negative samples: {len(negative_indices)}")

    rms_pos, rms_neg = _calculate_split_rms(
        positive_indices, negative_indices, hop_size
    )
    if debug:
        print(f"[DEBUG] rms_pos: {rms_pos}, rms_neg: {rms_neg}")

    if rms_pos is None or rms_neg is None:
        corner_positions = pad_to_expected_size(
            corner_positions,
            expected_size=feat_size,
            pad_value=0,
            contiguous=contiguous,
            debug=debug,
        )
        if debug:
            print("[DEBUG] max corners: 0")
            print("[DEBUG] min corners: 0")
            print(f"[DEBUG] corner_positions shape: {corner_positions.shape}")
            print(
                f"[DEBUG] corner_positions unique values: {np.unique(corner_positions)}"
            )
        return corner_positions

    max_start = None
    min_start = None
    max_count = 0
    min_count = 0
    min_peak_height = rms_pos * corner_position.peak_height_factor
    for i in range(1, len(audio)):
        prev = audio[i - 1]
        curr = audio[i]

        if prev < rms_pos and curr >= rms_pos:
            max_start = i - 1

        if prev > rms_pos and curr <= rms_pos:
            if max_start is None:
                continue
            max_end = i + 1
            max_region = audio[max_start:max_end]
            max_local_idx = np.argmax(max_region)
            max_idx = max_start + max_local_idx
            max_value = audio[max_idx]
            if max_value < min_peak_height:
                continue
            corner_positions[max_idx // hop_size] = 1.0
            max_count += 1

        if prev > rms_neg and curr <= rms_neg:
            min_start = i - 1

        if prev < rms_neg and curr >= rms_neg:
            if min_start is None:
                continue
            min_end = i + 1
            min_region = audio[min_start:min_end]
            min_local_idx = np.argmin(min_region)
            min_idx = min_start + min_local_idx
            min_value = audio[min_idx]
            if min_value > -min_peak_height:
                continue
            corner_positions[min_idx // hop_size] = -1.0
            min_count += 1

    corner_positions = pad_to_expected_size(
        corner_positions,
        expected_size=feat_size,
        pad_value=0,
        contiguous=contiguous,
        debug=debug,
    )
    if debug:
        print(f"[DEBUG] max corners: {max_count}")
        print(f"[DEBUG] min corners: {min_count}")
        print(f"[DEBUG] corner_positions shape: {corner_positions.shape}")
        print(f"[DEBUG] corner_positions unique values: {np.unique(corner_positions)}")
    return corner_positions
