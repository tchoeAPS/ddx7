from ddx7 import spectral_ops
from audio_utils import pad_to_expected_size
from ddx7.core import _DB_RANGE


def extract_f0(self, audio):
    f0, confidence = spectral_ops.calc_f0(
        audio,
        rate=self.sr,
        hop_size=self.hop_size,
        fmin=self.crepe_params.fmin,
        fmax=self.crepe_params.fmax,
        model=self.crepe_params.model,
        batch_size=self.crepe_params.batch_size,
        device=self.device,
        center=self.center,
    )

    if confidence.mean() < self.crepe_params.confidence_threshold:
        # print("Low confidence: {}".format(confidence.mean()))
        raise ValueError("Low f0 confidence")

    f0 = pad_to_expected_size(self, f0, expected_size=self.feat_size, pad_value=0)

    return f0


def calc_loudness(self, audio):
    loudness = spectral_ops.calc_loudness(
        audio,
        rate=self.sr,
        n_fft=self.loudness_params.nfft,
        hop_size=self.hop_size,
        center=self.center,
    )

    loudness = pad_to_expected_size(
        self, loudness, expected_size=self.feat_size, pad_value=-_DB_RANGE
    )
    return loudness


# TODO: Add center padding capability here.
def calc_rms(self, audio):
    rms = spectral_ops.calc_power(
        audio, frame_size=self.rms.frame_size, hop_size=self.hop_size, pad_end=True
    )
    rms = pad_to_expected_size(
        self, rms, expected_size=self.feat_size, pad_value=-_DB_RANGE
    )
    return rms
