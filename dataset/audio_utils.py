import operator
import functools
import numpy as np


def process_indices(self, indices: list) -> list:
    # Length in samples.
    max_len = self.max_len * self.sr

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
    new_indices = [x for x in new_indices if (x[1] - x[0] > self.seq_len * self.sr)]
    return new_indices


def pad_to_expected_size(self, features, expected_size, pad_value):

    # Pad to next integer division if we are processing a whole file in one go.
    if self.contiguous == True:
        # Pad up to next integer division
        pad_len = (
            features.shape[-1] // expected_size + 1
        ) * expected_size - features.shape[-1]
        # print(f'feat len {features.shape[-1]} expected {expected_size} pad {pad_len}')
        features = np.pad(features, (0, pad_len), "constant", constant_values=pad_value)
        return features
    else:
        if self.debug:
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
