from shutil import copyfile
import hydra
from pathlib import Path
import os
from functools import partial
from tqdm.contrib.concurrent import thread_map
from path_utils import ensure_nested_dir


def _create_mono_urmp(instrument_key, audio_files, target_dir, instruments_dict):
    target_dir = target_dir / instruments_dict[instrument_key]
    if not target_dir.exists():
        target_dir.mkdir(exist_ok=True)
    cur_audio_files = [
        audio_file
        for audio_file in audio_files
        if f"_{instrument_key}_" in audio_file.name
    ]
    [
        copyfile(audio_file, target_dir / audio_file.name)
        for audio_file in cur_audio_files
    ]


def _create_mono_testset(audio_files, target_dir, instrument):

    target_dir = target_dir / instrument
    if not target_dir.exists():
        target_dir.mkdir(exist_ok=True)
    cur_audio_files = [audio_file for audio_file in audio_files]
    [
        copyfile(audio_file, target_dir / audio_file.name)
        for audio_file in cur_audio_files
    ]


def make_testset(args):
    CWD = Path(hydra.utils.get_original_cwd())  # Get current directory
    os.chdir(CWD)

    if args.testset is not None:

        if args.skip_copy is False:

            # Create directories if needed
            target_dir = ensure_nested_dir(CWD, args.testset.input_dir)

            testset_path = CWD / args.testset.source_folder
            print("[INFO] Testset source path: {}".format(testset_path))
            print(
                "[INFO] will source files from directories: {}".format(
                    args.testset.instruments
                )
            )

            for instrument in args.testset.instruments:
                # print(instrument)
                # Find relevant audio files.
                test_audio_path = testset_path / instrument
                test_audio_files = list(test_audio_path.glob(f"./*.wav"))
                # print(test_audio_files)

                _create_mono_testset(
                    audio_files=test_audio_files,
                    target_dir=target_dir,
                    instrument=instrument,
                )

    if args.skip_process is False:

        # Create output dirs if needed
        ensure_nested_dir(CWD, args.testset.output_dir)

        # Process Test Set

        data_processor = hydra.utils.instantiate(args.data_processor)

        # Override original crepe confidence to process all the testset file.
        data_processor.set_confidence(0.0)
        data_processor.contiguous = args.testset.contiguous
        data_processor.contiguous_clip_noise = (
            args.testset.clip_noise
        )  # Clip frequencies tracked due to noise.

        data_processor.run_on_dirs(
            CWD / args.testset.input_dir, CWD / args.testset.output_dir
        )
    return


def make_urmp(args):
    # Phase 0 - copy all urmp wavs to corresponding folders
    CWD = Path(hydra.utils.get_original_cwd())  # Get current directory
    os.chdir(CWD)

    if args.urmp is not None:
        urmp_path = CWD / args.urmp.source_folder

        if args.skip_copy is False:

            # Create directories if needed
            target_dir = ensure_nested_dir(CWD, args.urmp.input_dir)

            # Find relevant audio files.
            urmp_audio_files = list(urmp_path.glob(f"./*/{args.urmp.mono_regex}*.wav"))

            print("[INFO] URMP Path: {}".format(urmp_path))

            print("[INFO] Number of files: {}".format(len(urmp_audio_files)))

            print(args.urmp.instruments.keys())
            # Partial function with instruments pre-configured for processing.
            create_mono_urmp_partial = partial(
                _create_mono_urmp,
                audio_files=urmp_audio_files,
                target_dir=target_dir,
                instruments_dict=args.urmp.instruments,
            )

            # Spawn threads to copy files.
            thread_map(create_mono_urmp_partial, list(args.urmp.instruments.keys()))

    # Process Train Set
    if args.skip_process is False:

        # Create output dirs if needed
        ensure_nested_dir(CWD, args.urmp.output_dir)

        data_processor = hydra.utils.instantiate(args.data_processor)

        data_processor.run_on_dirs(
            CWD / args.urmp.input_dir, CWD / args.urmp.output_dir
        )
    return
