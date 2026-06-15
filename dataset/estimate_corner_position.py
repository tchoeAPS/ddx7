from pathlib import Path
import librosa
import numpy as np
import functools
import operator
import matplotlib.pyplot as plt

def process_indices(indices: list, sr) -> list:
    # Length in samples.
    max_len = 4 * sr  # Assuming a sample rate of 16 kHz
    min_len = 3 * sr

    def expand_long(indices_tuple: tuple) -> list:
        if indices_tuple[1] - indices_tuple[0] > max_len:
            ret = [(start, start+max_len) for start in np.arange(indices_tuple[0], indices_tuple[1] - max_len, max_len)]
            ret.append((ret[-1][-1], min(ret[-1][-1] + max_len, indices_tuple[1])))
            return ret
        else:
            return [indices_tuple]

    new_indices = [*map(expand_long, indices)]
    new_indices = functools.reduce(operator.concat, new_indices, [])
    new_indices = [x for x in new_indices if (x[1] - x[0] >= min_len)]
    return new_indices

def split_positive_negative_indices(indices: list, sr: int) -> tuple:
    positive_indices = []
    negative_indices = []
    for i, val in enumerate(indices):
        if val > 0:
            positive_indices.append([i/sr, val])
        elif val < 0:
            negative_indices.append([i/sr, val])
    return positive_indices, negative_indices

# def regression_lines

def main():
    wav_path = Path("files/train/violin/AuSep_1_vn_01_Jupiter.wav")

    audio, sr = librosa.load(wav_path, sr=16000)
    sound_indices = librosa.effects.split(audio, top_db=40)
    sound_indices = process_indices(sound_indices, sr)
    audio_chunk = audio[sound_indices[0][0]:sound_indices[0][1]]
    split_indices = split_positive_negative_indices(audio_chunk, sr)

    # 1. Calculate RMS+ and RMS-
    y_pos_vals = [x[1] for x in split_indices[0]]
    y_neg_vals = [x[1] for x in split_indices[1]]
    rms_pos = librosa.feature.rms(y=np.array(y_pos_vals), frame_length=len(split_indices[0]), hop_length=64, center=False)
    rms_neg = librosa.feature.rms(y=np.array(y_neg_vals), frame_length=len(split_indices[1]), hop_length=64, center=False)
    print(f"RMS+: {rms_pos}, RMS-: {rms_neg}")

    # 2. Find where the waveform crosses RMS- and RMS+
    rms_pos = rms_pos[0, 0]
    rms_neg = -rms_neg[0, 0]
    corners = []
    peak_start = None
    peak_end = None
    trough_start = None
    trough_end = None
    min_peak_height = rms_pos * 1.5
    for i in range(1, len(audio_chunk)):
        prev = audio_chunk[i - 1]
        curr = audio_chunk[i]

        if prev < rms_pos and curr >= rms_pos:
            peak_start = i - 1

        if prev > rms_pos and curr <= rms_pos:
            peak_end = i + 1
            region = audio_chunk[peak_start:peak_end]
            peak_local_idx = np.argmax(region)
            peak_idx = peak_start + peak_local_idx
            peak_value = audio_chunk[peak_idx]
            if peak_value < min_peak_height:
                continue
            corners.append([i / sr, audio_chunk[peak_idx]])
        
        if prev > rms_neg and curr <= rms_neg:
            trough_start = i - 1
        
        if prev < rms_neg and curr >= rms_neg:
            if trough_start is None:
                continue
            trough_end = i + 1
            region = audio_chunk[trough_start:trough_end]
            peak_local_idx = np.argmin(region)
            peak_idx = trough_start + peak_local_idx
            peak_value = audio_chunk[peak_idx]
            if peak_value > -min_peak_height:
                continue
            corners.append([i / sr, audio_chunk[peak_idx]])

    # X/Y values for the measured audio waveform.
    t = np.arange(len(audio_chunk)) / sr
    y = audio_chunk

    plt.figure()
    # Blue line: measured audio/displacement.
    plt.plot(t, y, label="audio")

    # Red dot: estimated corner from the intersection of the two fitted lines.
    peak_times = [corner[0] for corner in corners]
    peak_values = [corner[1] for corner in corners]
    plt.scatter(
        peak_times,
        peak_values,
        color="red",
        zorder=5,
    )

    # Dashed blue lines: RMS thresholds used to choose segment boundaries.
    plt.axhline(rms_pos, linestyle="--", color="blue", label="RMS+")
    plt.axhline(rms_neg, linestyle="--", color="blue", label="RMS-")

    # plt.title(f"{corner['type']} valid={corner['valid']}")
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement")
    plt.legend()
    plt.show()
    return

if __name__ == "__main__":
    main()
