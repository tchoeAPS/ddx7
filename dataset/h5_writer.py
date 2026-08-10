import h5py


def save_data(audio, f0, loudness, rms, h5f, counter, debug, corner_positions=None, corner_duty_cycle=None):
    h5f.create_dataset(f"{counter}_audio", data=audio)
    h5f.create_dataset(f"{counter}_f0", data=f0)
    h5f.create_dataset(f"{counter}_loudness", data=loudness)
    h5f.create_dataset(f"{counter}_rms", data=rms)
    if corner_positions is not None:
        h5f.create_dataset(f"{counter}_corner_position", data=corner_positions)
        if debug:
            print(
                f"[DEBUG] saved {counter}_corner_position shape: {corner_positions.shape}"
            )
    if corner_duty_cycle is not None:
        h5f.create_dataset(f"{counter}_corner_duty_cycle", data=corner_duty_cycle)
        if debug:
            print(
                f"[DEBUG] saved {counter}_corner_duty_cycle shape: {corner_duty_cycle.shape}"
            )
    return counter + 1


def init_h5(sr, data_dir):
    return h5py.File(data_dir / f"{sr}.h5", "w")


def close_h5(h5f):
    h5f.close()
