import mne
import numpy as np
import h5py
from scipy.io import loadmat

# The trigger number (keys) and song number (values)
trigger_to_song = {
    "DI21": 1,
    "DI22": 2,
    "DI23": 3,
    "DI24": 4,
    "DI25": 5,
    "DI26": 6,
    "DI27": 7,
    "DI28": 8,
    "DI29": 9,
    "DI30": 10
}

# Duration of songs in seconds
song_durations = {
    "DI21": 4*60 + 38,
    "DI22": 4*60 + 31,
    "DI23": 4*60 + 36,
    "DI24": 4*60 + 54,
    "DI25": 4*60 + 49,
    "DI26": 4*60 + 36,
    "DI27": 4*60 + 52,
    "DI28": 4*60 + 52,
    "DI29": 4*60 + 54,
    "DI30": 4*60 + 58
}

def decode_hdf5_value(obj):
    """Decode matlab datatypes to python compatable"""
    data = obj[()]
    # MATLAB char/string stored as uint16
    if data.dtype == np.uint16:
        return ''.join(chr(x) for x in data.flatten())

    # Numeric scalar
    if data.size == 1:
        return data.item()
    print(data.dtype)
    return data

def load_raw(file_path : str):
    """
    Loads the raw EEG data from a .mat file.
    Args:
        file_path (str): The path to the file.
    Returns:
        raw (mne.RawArray): A RawArray consisting of the raw EEG data and channels.
    """
    with h5py.File(file_path, "r") as f:
        # EEG data
        X = np.array(f["X"])

        X = 0.1 * X.astype(float)
        # Transpose to: (channels, samples)
        X = (X * 1e-6).T
        # Sampling frequency
        fs = int(np.array(f["fs"]).squeeze())

        ch_names = [f"E{i}" for i in range(1, 130)]

        info = mne.create_info(
            ch_names=ch_names,
            sfreq=fs,
            ch_types="eeg"
        )

        raw = mne.io.RawArray(X, info)

        din = f["DIN_1"]
        # Decoded 46 x 4 list
        # [trigger label, event sample, 1.0, same event sample]
        din_cells = []
        for i in range(din.shape[0]):
            row = []
            for j in range(din.shape[1]):
                ref = din[i, j]
                obj = f[ref]
                value = decode_hdf5_value(obj)
                row.append(value)
            din_cells.append(row)

        # List of trigger codes and samples to identify song onsets
        trigger_codes = []
        trigger_samples = []
        for row in din_cells:
            code = row[0]
            sample = row[1]

            # Only add song codes and samples
            for key in trigger_to_song:
                if key == code:
                     trigger_codes.append(code)
                     trigger_samples.append(sample)

        # Annotations containing song number, song duration, and onset time (s)
        annotations = []
        for trig, start_sample in zip(trigger_codes, trigger_samples):
            onset_sec = start_sample / fs
            duration_sec = song_durations[trig]
            song_num = trigger_to_song[trig]

            annotations.append(
                mne.Annotations(
                    onset=[onset_sec],
                    duration=[duration_sec],
                    description=[f"song_{song_num}"]
                )
            )

        raw.set_annotations(annotations[0])
        for ann in annotations[1:]:
            raw.set_annotations(raw.annotations + ann)
        return raw


def load_preprocessed(file_path : str, subject_id : int, song_id: int):
    """
    Loads the preprocessed EEG data for a subject.
    Args:
        file_path (str): The path to the file.
        subject_id (int): The subject ID.
        song_id (int) : The song number.
    Returns:
        raw (mne.RawArray): A RawArray consisting of the preprocessed
         EEG data and channels.
    """
    file = loadmat(file_path)
    data = file[f"data{song_id}"].squeeze()
    data = data.transpose(2, 0, 1)
    X = data[subject_id]
    X = 0.1 * X.astype(float)
    X = (X * 1e-6)
    ch_names = [f"E{i}" for i in range(1, 126)]
    fs = int(file["fs"].squeeze())

    info = mne.create_info(
        ch_names=ch_names,
        sfreq=fs,
        ch_types="eeg"
    )

    raw = mne.io.RawArray(X, info)

    return raw


def dataset_to_numpy(dataset):
    eeg_list = []
    audio_list = []

    for i in range(len(dataset)):
        sample = dataset[i]

        eeg_list.append(sample["eeg"])
        audio_list.append(sample["audio"])

    eeg = np.stack(eeg_list)
    audio = np.stack(audio_list)

    return eeg, audio
