from pathlib import Path
import librosa
import numpy as np
import functools
import operator
import hydra

def process_indices(indices: list, sr, min_len_seconds: int, max_len_seconds: int) -> list:
    # Length in samples.
    max_len = max_len_seconds * sr
    min_len = min_len_seconds * sr

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

def get_default_wav_path(args) -> Path:
    cwd = Path(hydra.utils.get_original_cwd())
    instrument = next(iter(args.urmp.instruments.values()))
    input_dir = cwd / args.urmp.input_dir / instrument
    wav_paths = sorted(input_dir.glob("*.wav"))
    for wav_path in wav_paths:
        return wav_path
    raise FileNotFoundError(f"No .wav files found in {input_dir}")


def calculate_split_rms(positive_indices: list, negative_indices: list, hop_size: int = 64) -> tuple:
    y_pos_vals = [x[1] for x in positive_indices]
    y_neg_vals = [x[1] for x in negative_indices]
    rms_pos = librosa.feature.rms(y=np.array(y_pos_vals), frame_length=len(positive_indices), hop_length=hop_size, center=False)
    rms_neg = librosa.feature.rms(y=np.array(y_neg_vals), frame_length=len(negative_indices), hop_length=hop_size, center=False)
    print(f"RMS+: {rms_pos}, RMS-: {rms_neg}")
    return rms_pos[0, 0], -rms_neg[0, 0]

def find_corners(audio_chunk, sr, rms_pos, rms_neg, peak_height_factor):
    corners = []
    corner_positions = [0] * len(audio_chunk)
    max_start = None
    max_end = None
    min_start = None
    min_end = None
    min_peak_height = rms_pos * peak_height_factor
    for i in range(1, len(audio_chunk)):
        prev = audio_chunk[i - 1]
        curr = audio_chunk[i]

        # start of max
        if prev < rms_pos and curr >= rms_pos:
            max_start = i - 1

        # end of max
        if prev > rms_pos and curr <= rms_pos:
            if max_start is None:
                continue
            max_end = i + 1
            max_region = audio_chunk[max_start:max_end]
            max_local_idx = np.argmax(max_region)
            max_idx = max_start + max_local_idx
            max_value = audio_chunk[max_idx]
            if max_value < min_peak_height:
                continue
            corners.append([i / sr, audio_chunk[max_idx]])
            corner_positions[i] = 1
        
        # start of min
        if prev > rms_neg and curr <= rms_neg:
            min_start = i - 1
        
        # end of min
        if prev < rms_neg and curr >= rms_neg:
            if min_start is None:
                continue
            min_end = i + 1
            min_region = audio_chunk[min_start:min_end]
            min_local_idx = np.argmin(min_region)
            min_idx = min_start + min_local_idx
            min_value = audio_chunk[min_idx]
            if min_value > -min_peak_height:
                continue
            corners.append([i / sr, audio_chunk[min_idx]])
            corner_positions[i] = 1
    return corners

@hydra.main(config_path="./", config_name="data_config.yaml", version_base=None)
def main(args):
    data_processor = args.data_processor
    wav_path = get_default_wav_path(args)

    audio, sr = librosa.load(wav_path, sr=data_processor.sr)
    sound_indices = librosa.effects.split(audio, top_db=data_processor.silence_thresh_dB)
    sound_indices = process_indices(
        sound_indices,
        sr,
        min_len_seconds=data_processor.seq_len,
        max_len_seconds=data_processor.max_len,
    )
    audio_chunk = audio[sound_indices[0][0]:sound_indices[0][1]]
    positive_indices, negative_indices = split_positive_negative_indices(audio_chunk, sr)

    # 1. Calculate RMS+ and RMS-
    rms_pos, rms_neg = calculate_split_rms(
        positive_indices,
        negative_indices,
        hop_size=data_processor.hop_size,
    )

    # 2. Find where the waveform crosses RMS- and RMS+
    corners = find_corners(
        audio_chunk,
        sr,
        rms_pos,
        rms_neg,
        peak_height_factor=data_processor.corner_position.peak_height_factor,
    )

    # X/Y values for the measured audio waveform.
    t = np.arange(len(audio_chunk)) / sr
    y = audio_chunk

    import matplotlib.pyplot as plt

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
