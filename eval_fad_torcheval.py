"""
Cross-check eval_fad.py's FAD numbers using a second, independent implementation:
torcheval's FrechetAudioDistance (TorchAudio VGGish embeddings), instead of the
gudgud96/frechet-audio-distance package used in eval_fad.py.

Reuses the wav directories already written by a prior `python eval_fad.py` run:
    <instrument background>/fad_background
    <run_dir>/<exp_name>/<run_name>/fad_eval/test_real
    <run_dir>/<exp_name>/<run_name>/fad_eval/generated

Usage:
    python eval_fad_torcheval.py runs/corner_duty_cycle/seed1
"""
import sys
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from torcheval.metrics.audio.fad import FrechetAudioDistance

# torchaudio 2.8.0's deprecated VGGish prototype module references `torch`/
# `torchaudio` as globals without importing them itself (upstream bug) -
# inject them so VGGishBundle._get_state_dict() doesn't NameError.
import torchaudio.prototype.pipelines._vggish._vggish_pipeline as _vggish_pipeline_mod

_vggish_pipeline_mod.torch = torch
_vggish_pipeline_mod.torchaudio = torchaudio


def load_wavs(directory):
    waves = []
    for p in sorted(Path(directory).glob("*.wav")):
        audio, sr = sf.read(p, dtype="float32")
        assert sr == 16000, f"{p} has sr={sr}, expected 16000"
        waves.append(torch.from_numpy(audio))
    return waves


def score(background_dir, eval_dir, device):
    fad = FrechetAudioDistance.with_vggish(device=device)
    empty = torch.zeros(0, 1)
    for wave in load_wavs(background_dir):
        fad.update(preds=empty, targets=wave.unsqueeze(0))
    for wave in load_wavs(eval_dir):
        fad.update(preds=wave.unsqueeze(0), targets=empty)
    return fad.compute().item()


def main():
    run_dir = Path(sys.argv[1])
    background_dir = Path("dataset/data/train/violin/fad_background")
    fad_eval_dir = run_dir / "fad_eval"
    device = torch.device("cpu")

    test_data_fad = score(background_dir, fad_eval_dir / "test_real", device)
    print(f"[RESULT] Test Data FAD (torcheval/vggish): {test_data_fad}")

    model_fad = score(background_dir, fad_eval_dir / "generated", device)
    print(f"[RESULT] Model FAD (torcheval/vggish): {model_fad}")


if __name__ == "__main__":
    main()
