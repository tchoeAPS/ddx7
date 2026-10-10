from ddx7 import spectral_ops
from audio_utils import pad_to_expected_size
from ddx7.core import _DB_RANGE


def extract_f0(
    audio, sr, hop_size, crepe_params, device, center, feat_size, contiguous, debug
):
    f0, confidence = spectral_ops.calc_f0(
        audio,
        rate=sr,
        hop_size=hop_size,
        fmin=crepe_params.fmin,
        fmax=crepe_params.fmax,
        model=crepe_params.model,
        batch_size=crepe_params.batch_size,
        device=device,
        center=center,
    )

    if confidence.mean() < crepe_params.confidence_threshold:
        # print("Low confidence: {}".format(confidence.mean()))
        raise ValueError("Low f0 confidence")

    f0 = pad_to_expected_size(
        f0,
        expected_size=feat_size,
        pad_value=0,
        contiguous=contiguous,
        debug=debug,
    )

    return f0


def calc_loudness(
    audio, sr, loudness_params, hop_size, center, feat_size, contiguous, debug
):
    loudness = spectral_ops.calc_loudness(
        audio,
        rate=sr,
        n_fft=loudness_params.nfft,
        hop_size=hop_size,
        center=center,
    )

    loudness = pad_to_expected_size(
        loudness,
        expected_size=feat_size,
        pad_value=-_DB_RANGE,
        contiguous=contiguous,
        debug=debug,
    )
    return loudness


# TODO: Add center padding capability here.
def calc_rms(audio, rms, hop_size, feat_size, contiguous, debug):
    rms = spectral_ops.calc_power(
        audio, frame_size=rms.frame_size, hop_size=hop_size, pad_end=True
    )
    rms = pad_to_expected_size(
        rms,
        expected_size=feat_size,
        pad_value=-_DB_RANGE,
        contiguous=contiguous,
        debug=debug,
    )

    return rms
