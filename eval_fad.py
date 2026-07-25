"""
Computes Frechet Audio Distance (FAD) between reference and synthesized
audio for a trained model's test set.

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

    model = hydra.utils.instantiate(args.model)
    checkpoint = args.get("checkpoint", "state_best.pth")
    ckpt_path = f"{root}/{args.run_dir}/{args.exp_name}/{args.run_name}/{checkpoint}"
    model.load_state_dict(torch.load(ckpt_path, map_location=args.device))
    model = model.to(args.device).eval()

    preprocessor = F0LoudnessRMSPreprocessor()

    out_dir = f"{root}/{args.run_dir}/{args.exp_name}/{args.run_name}/fad_eval"
    ref_dir = os.path.join(out_dir, "reference")
    gen_dir = os.path.join(out_dir, "generated")
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(gen_dir, exist_ok=True)

    sr = model.get_sr()
    with torch.no_grad():
        for i, x in enumerate(test_loader):
            x = preprocessor.run(x)
            synth_out = model(x)
            ref_audio = x["audio"].squeeze().cpu().numpy()
            gen_audio = synth_out["synth_audio"].squeeze().cpu().numpy()
            sf.write(os.path.join(ref_dir, f"{i:04d}.wav"), ref_audio, sr)
            sf.write(os.path.join(gen_dir, f"{i:04d}.wav"), gen_audio, sr)

    print(f"[INFO] Wrote {len(test_loader)} reference/generated pairs to {out_dir}")

    frechet = FrechetAudioDistance(
        model_name="vggish",
        sample_rate=16000,
        use_pca=False,
        use_activation=False,
        verbose=True,
    )
    fad_score = frechet.score(ref_dir, gen_dir)
    print(f"[RESULT] FAD score: {fad_score}")


if __name__ == "__main__":
    main()
