"""
Computes Frechet Audio Distance (FAD), following the same procedure as the
DDX7 paper (Section 5.2.1): a "background" embedding distribution is built
from the COMPLETE audio corpus of the instrument (not just the test split),
and FAD is computed between that background and (a) the model's resynthesized
test set, and (b) the real test set itself (a reference floor, matching the
paper's "Test Data" row).

Usage:
    python eval_fad.py model=tcnres_bow_fmstr exp_name=bow_experiment run_name=run1

Optional overrides:
    checkpoint=state_best.pth   (default: state_best.pth)
"""
import os

import hydra
import soundfile as sf
import torch
from frechet_audio_distance import FrechetAudioDistance
from torch.utils.data import DataLoader, random_split

from ddx7.data_utils.h5_dataset import h5Dataset
from ddx7.data_utils.preprocessor import F0LoudnessRMSPreprocessor


def write_wavs(loader, out_dir, sr, key):
    if os.path.isdir(out_dir) and len(os.listdir(out_dir)) > 0:
        print(f"[INFO] {out_dir} already populated, skipping.")
        return
    os.makedirs(out_dir, exist_ok=True)
    for i, x in enumerate(loader):
        audio = x[key].squeeze().cpu().numpy()
        sf.write(os.path.join(out_dir, f"{i:04d}.wav"), audio, sr)
    print(f"[INFO] Wrote {len(loader)} wavs to {out_dir}")


@hydra.main(config_path="recipes", config_name="config.yaml", version_base=None)
def main(args):
    torch.manual_seed(args.seed)
    root = hydra.utils.get_original_cwd()

    # Rebuild the same train/valid/test split used during training (same seed -> same split).
    input_keys = ("audio", "loudness", "f0", "rms", "corner_position")
    data_path = f"{root}/{args.data_dir}/train/{args.instrument}/16000.h5"
    dataset = h5Dataset(
        sr=16000,
        data_path=data_path,
        input_keys=input_keys,
        max_audio_val=1,
        device=args.device,
    )
    train_split = int(args.train_split * len(dataset))
    test_split = (len(dataset) - train_split) // 2
    val_split = len(dataset) - train_split - test_split
    _, _, test_set = random_split(dataset, [train_split, val_split, test_split])
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    full_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = hydra.utils.instantiate(args.model)
    checkpoint = args.get("checkpoint", "state_best.pth")
    ckpt_path = f"{root}/{args.run_dir}/{args.exp_name}/{args.run_name}/{checkpoint}"
    model.load_state_dict(torch.load(ckpt_path, map_location=args.device))
    model = model.to(args.device).eval()

    preprocessor = F0LoudnessRMSPreprocessor()
    sr = model.get_sr()

    # Background: complete audio corpus for the instrument (shared across runs/models).
    background_dir = f"{root}/{args.data_dir}/train/{args.instrument}/fad_background"
    write_wavs(full_loader, background_dir, sr, "audio")

    # Per-run outputs: real test-set audio (the paper's "Test Data" comparison)
    # and the model's resynthesized test set.
    out_dir = f"{root}/{args.run_dir}/{args.exp_name}/{args.run_name}/fad_eval"
    test_real_dir = os.path.join(out_dir, "test_real")
    gen_dir = os.path.join(out_dir, "generated")
    os.makedirs(test_real_dir, exist_ok=True)
    os.makedirs(gen_dir, exist_ok=True)

    write_wavs(test_loader, test_real_dir, sr, "audio")

    with torch.no_grad():
        for i, x in enumerate(test_loader):
            x = preprocessor.run(x)
            synth_out = model(x)
            gen_audio = synth_out["synth_audio"].squeeze().cpu().numpy()
            sf.write(os.path.join(gen_dir, f"{i:04d}.wav"), gen_audio, sr)
    print(f"[INFO] Wrote {len(test_loader)} generated wavs to {gen_dir}")

    frechet = FrechetAudioDistance(
        model_name="vggish",
        sample_rate=16000,
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
