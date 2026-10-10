"""
Computes Frechet Audio Distance (FAD), following the same procedure as the
DDX7 paper (Section 5.2.1): a "background" embedding distribution is built
from the COMPLETE audio corpus of the instrument (not just the test split),
and FAD is computed between that background and (a) the model's resynthesized
test set, and (b) the real test set itself (a reference floor, matching the
paper's "Test Data" row).

Reads real/generated test-set audio straight out of test.h5 (written by
trainer.py's post-training test() pass over the best checkpoint), so this
never re-instantiates the model or reloads the checkpoint.

Usage:
    python eval_fad.py runs/corner_duty_cycle/seed1
"""
import os
import sys

import h5py
import soundfile as sf
import yaml

# frechet_audio_distance drags in laion_clap, which calls argparse.parse_args()
# at import time (even though we only use the vggish backend, never CLAP) -
# hiding sys.argv here stops it from choking on our own positional arg.
_argv = sys.argv
sys.argv = [sys.argv[0]]
from frechet_audio_distance import FrechetAudioDistance
sys.argv = _argv

SR = 16000


def write_wavs_from_h5(h5_path, key_suffix, out_dir):
    """Write each `{i}{key_suffix}` dataset in an h5 file to its own wav."""
    if os.path.isdir(out_dir) and len(os.listdir(out_dir)) > 0:
        print(f"[INFO] {out_dir} already populated, skipping.")
        return
    os.makedirs(out_dir, exist_ok=True)
    with h5py.File(h5_path, "r") as f:
        keys = sorted(
            (k for k in f.keys() if k.endswith(key_suffix)),
            key=lambda k: int(k[: -len(key_suffix)]),
        )
        for i, k in enumerate(keys):
            sf.write(os.path.join(out_dir, f"{i:04d}.wav"), f[k][:], SR)
    print(f"[INFO] Wrote {len(keys)} wavs to {out_dir}")


def write_wavs_from_flat_array(h5_path, array_key, n_clips, out_dir):
    """Write a flat concatenated array (test.h5's audio/synth_audio) as separate wavs."""
    if os.path.isdir(out_dir) and len(os.listdir(out_dir)) > 0:
        print(f"[INFO] {out_dir} already populated, skipping.")
        return
    os.makedirs(out_dir, exist_ok=True)
    with h5py.File(h5_path, "r") as f:
        audio = f[array_key][:]
    clip_len = audio.size // n_clips
    for i in range(n_clips):
        clip = audio[i * clip_len : (i + 1) * clip_len]
        sf.write(os.path.join(out_dir, f"{i:04d}.wav"), clip, SR)
    print(f"[INFO] Wrote {n_clips} wavs to {out_dir}")


def main():
    run_dir = sys.argv[1]

    with open(os.path.join(run_dir, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    instrument = cfg["instrument"]
    data_dir = cfg["data_dir"]

    test_h5 = os.path.join(run_dir, "test.h5")
    with h5py.File(test_h5, "r") as f:
        n_clips = f["f0"].shape[0]

    background_dir = os.path.join(data_dir, "train", instrument, "fad_background")
    write_wavs_from_h5(
        os.path.join(data_dir, "train", instrument, "16000.h5"), "_audio", background_dir
    )

    out_dir = os.path.join(run_dir, "fad_eval")
    test_real_dir = os.path.join(out_dir, "test_real")
    gen_dir = os.path.join(out_dir, "generated")
    write_wavs_from_flat_array(test_h5, "audio", n_clips, test_real_dir)
    write_wavs_from_flat_array(test_h5, "synth_audio", n_clips, gen_dir)

    frechet = FrechetAudioDistance(
        model_name="vggish",
        sample_rate=SR,
        use_pca=False,
        use_activation=False,
        verbose=True,
    )

    test_data_fad = frechet.score(background_dir, test_real_dir)
    print(f"[RESULT] Test Data FAD (real test set vs. background): {test_data_fad}")

    model_fad = frechet.score(background_dir, gen_dir)
    print(f"[RESULT] Model FAD (generated test set vs. background): {model_fad}")


if __name__ == "__main__":
    main()
