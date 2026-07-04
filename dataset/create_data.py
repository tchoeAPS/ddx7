from tqdm import tqdm
import hydra
from pathlib import Path
import torch
from audio_utils import (
    pad_to_expected_size,
    load_normalized_audio,
    get_sound_indices,
)
from estimate_corner_position import calc_corner_positions
from features import extract_f0, calc_loudness, calc_rms
from h5_writer import save_data, init_h5, close_h5
from ddx7.core import _DB_RANGE
from prepare_sources import make_testset, make_urmp

"""
URMP Data processor class adapted from HTP paper.
https://github.com/mosheman5/timbre_painting

"""


class ProcessData:
    def __init__(
        self,
        silence_thresh_dB,
        sr,
        device,
        seq_len,
        crepe_params,
        loudness_params,
        rms_params,
        hop_size,
        max_len,
        center,
        corner_position=None,
        overlap=0.0,
        debug=False,
        contiguous=False,
        contiguous_clip_noise=False,
    ):
        super().__init__()
        self.silence_thresh_dB = silence_thresh_dB
        self.crepe_params = crepe_params
        self.sr = sr
        self.device = torch.device(device)
        self.seq_len = seq_len
        self.loudness_params = loudness_params
        self.rms = rms_params
        self.max_len = max_len
        self.hop_size = hop_size
        self.feat_size = self.max_len * self.sr // self.hop_size
        self.audio_size = self.max_len * self.sr
        self.center = center
        self.corner_position = corner_position
        self.overlap = overlap
        self.debug = debug
        self.contiguous = contiguous
        self.contiguous_clip_noise = contiguous_clip_noise

    def set_confidence(self, confidence):
        self.crepe_params.confidence_threshold = confidence

    """
    Main audio processing function
    """

    def run_on_files(self, data_dir, input_dir, output_dir):
        audio_files = list((input_dir / data_dir).glob("*.wav"))
        output_dir = output_dir / data_dir
        output_dir.mkdir(exist_ok=True)

        # Open container
        h5f = init_h5(self, output_dir)
        counter = 0

        for audio_file in tqdm(audio_files):
            if self.debug:
                print("Processing: {}".format(audio_file))

            # load and split files
            data = load_normalized_audio(audio_file, self.sr)
            sounds_indices = get_sound_indices(
                data,
                silence_thresh_dB=self.silence_thresh_dB,
                contiguous=self.contiguous,
                processor=self,
            )
            if len(sounds_indices) == 0:
                continue

            for indices in sounds_indices:
                audio = data[indices[0] : indices[1]]
                if self.debug:
                    print(
                        "\tIndexes: {} {} - len: {}".format(
                            indices[0], indices[1], indices[1] - indices[0]
                        )
                    )

                # Feature retrieval segment

                try:  # Only process audio with enough CREPE confidence
                    f0 = extract_f0(self, audio)
                except ValueError:
                    continue

                # Further downsamples the audio back to the other specified sample rates and returns a dictionary.
                loudness = calc_loudness(self, audio)
                rms = calc_rms(self, audio)
                corner_positions = None
                corner_position_enabled = (
                    self.corner_position is not None
                    and self.corner_position.enabled is True
                )
                if self.debug:
                    print(f"[DEBUG] corner_position enabled: {corner_position_enabled}")
                if corner_position_enabled:
                    corner_positions = calc_corner_positions(self, audio)
                if self.contiguous:
                    if self.contiguous_clip_noise:
                        if self.debug:
                            print("[DEBUG] clipping noise")
                        clip_pos = f0 > 1900.0
                        loudness[clip_pos] = -_DB_RANGE
                    audio = pad_to_expected_size(
                        self, audio, f0.shape[0] * self.hop_size, 0
                    )

                else:
                    audio = pad_to_expected_size(self, audio, self.audio_size, 0)
                if self.debug:
                    print(
                        f"\t Store block {counter}: f0 : {f0.shape} - loudness : {loudness.shape} - rms {rms.shape} - audio : {audio.shape}"
                    )
                counter = save_data(
                    self,
                    audio,
                    f0,
                    loudness,
                    rms,
                    h5f,
                    counter,
                    corner_positions=corner_positions,
                )
                print(
                    f"[DEBUG] saved data block {counter - 1}: audio={audio}, f0={f0}, loudness={loudness}, rms={rms}, corner_positions={corner_positions}"
                )
                raise RuntimeError("Temporary stop after first saved data block")

        # Finished storing f0 and loudness
        close_h5(h5f)

    def run_on_dirs(self, input_dir: Path, output_dir: Path):
        # print("Starting with crepe confidence: {}".format(self.crepe_params.confidence_threshold))
        folders = [x for x in input_dir.glob("./*") if x.is_dir()]
        for folder in tqdm(folders):
            self.run_on_files(folder.name, input_dir, output_dir)


@hydra.main(config_path="./", config_name="data_config.yaml", version_base=None)
def main(args):

    if args.process_testset is True:
        make_testset(args)

    if args.process_urmp is True:
        make_urmp(args)


if __name__ == "__main__":
    main()
