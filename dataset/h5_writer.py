import h5py


def save_data(self, audio, f0, loudness, rms, h5f, counter, corner_positions=None):
    h5f.create_dataset(f"{counter}_audio", data=audio)
    h5f.create_dataset(f"{counter}_f0", data=f0)
    h5f.create_dataset(f"{counter}_loudness", data=loudness)
    h5f.create_dataset(f"{counter}_rms", data=rms)
    if corner_positions is not None:
        h5f.create_dataset(f"{counter}_corner_position", data=corner_positions)
        if self.debug:
            print(
                f"[DEBUG] saved {counter}_corner_position shape: {corner_positions.shape}"
            )
    return counter + 1


def init_h5(self, data_dir):
    return h5py.File(data_dir / f"{self.sr}.h5", "w")


def close_h5(h5f):
    h5f.close()
