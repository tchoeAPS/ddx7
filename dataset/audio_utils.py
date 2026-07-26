import operator
import functools
import numpy as np
import librosa


def _process_indices(max_len, sr, seq_len, indices: list) -> list:
    # Length in samples.
    max_len = max_len * sr

    def expand_long(indices_tuple: tuple) -> list:
        if indices_tuple[1] - indices_tuple[0] > max_len:
            ret = [
                (start, start + max_len)
                for start in np.arange(
                    indices_tuple[0], indices_tuple[1] - max_len, max_len
                )
            ]
            ret.append((ret[-1][-1], min(ret[-1][-1] + max_len, indices_tuple[1])))
            return ret
        else:
            return [indices_tuple]

    new_indices = [*map(expand_long, indices)]
    new_indices = functools.reduce(operator.concat, new_indices, [])
    new_indices = [x for x in new_indices if (x[1] - x[0] > seq_len * sr)]
    return new_indices


def pad_to_expected_size(features, expected_size, pad_value, contiguous, debug):

    # Pad to next integer division if we are processing a whole file in one go.
    if contiguous == True:
        # Pad up to next integer division
        pad_len = (
            features.shape[-1] // expected_size + 1
        ) * expected_size - features.shape[-1]
        # print(f'feat len {features.shape[-1]} expected {expected_size} pad {pad_len}')
        features = np.pad(features, (0, pad_len), "constant", constant_values=pad_value)
        return features
    else:
        if debug:
            print(
                "Feat shape {} - expected size: {}".format(
                    features.shape[-1], expected_size
                )
            )
        if features.shape[-1] < expected_size:
            pad_len = expected_size - features.shape[-1]
            features = np.pad(
                features, (0, pad_len), "constant", constant_values=pad_value
            )
        if features.shape[-1] > expected_size:
            raise Exception("Expected size is smaller than current value")
    return features


def load_normalized_audio(audio_file, sr):
    data, _ = librosa.load(audio_file.as_posix(), sr=sr)
    data = librosa.util.normalize(data)  # Peak-normalize audio
    return data


def get_sound_indices(audio, silence_thresh_dB, contiguous, max_len, sr, seq_len):
    if contiguous:
        return [[0, len(audio)]]

    # print("[DEBUG] Sound indices {}".format(sounds_indices))
    sound_indices = librosa.effects.split(audio, top_db=silence_thresh_dB)
    return _process_indices(max_len, sr, seq_len, sound_indices)
