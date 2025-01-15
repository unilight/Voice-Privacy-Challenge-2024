# This code is based on
# https://github.com/speechbrain/speechbrain/blob/develop/recipes/VoxCeleb/voxceleb_prepare.py
import csv
import os
import logging
import random
from pathlib import Path
import numpy as np
import torch
import torchaudio
import soundfile as sf
from tqdm.contrib import tqdm
from speechbrain.dataio.dataio import (
    load_pkl,
    save_pkl,
)

from utils import read_kaldi_format

logger = logging.getLogger(__name__)
OPT_FILE = "opt_jtube_prepare.pkl"
TRAIN_CSV = "train.csv"
DEV_CSV = "dev.csv"
ENROL_CSV = "enrol.csv"
SAMPLERATE = 16000

def prepare_jtube(
    data_folder,
    save_folder,
    splits=["train", "dev"],
    split_ratio=[90, 10],
    seg_dur=3.0,
    amp_th=5e-04,
    num_utt=None,
    num_spk=None,
    random_segment=False,
    skip_prep=False,
    utt_selected_ways="spk-random",
):
    """
    Prepares the csv files for the libri datasets.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original libri  dataset is stored.
    save_folder : str
        The directory where to store the csv files.
    verification_pairs_file : str
        txt file containing the verification split.
    splits : list
        List of splits to prepare from ['train', 'dev']
    split_ratio : list
        List if int for train and validation splits
    seg_dur : int
        Segment duration of a chunk in seconds (e.g., 3.0 seconds).
    amp_th : float
        removes segments whose average amplitude is below the
        given threshold.
    source : str
        Path to the folder where the VoxCeleb dataset source is stored.
    num_utt: float
        How many utterances for each speaker used for training
    num_spk: float
        How many speakers used for training
    random_segment : bool
        Train random segments
    skip_prep: Bool
        If True, skip preparation.

    Example
    -------
    >>> from libri_prepare import prepare_libri
    >>> data_folder = 'LibriSpeech/train-clean-360/'
    >>> save_folder = 'libri/'
    >>> splits = ['train', 'dev']
    >>> split_ratio = [90, 10]
    >>> prepare_libri(data_folder, save_folder, splits, split_ratio)
    """

    if skip_prep:
        return

    save_folder = Path(save_folder)
    save_folder.mkdir(exist_ok=True, parents=True)

    # Setting ouput files
    save_opt = save_folder / OPT_FILE
    save_csv_train = save_folder / TRAIN_CSV
    save_csv_dev = save_folder / DEV_CSV

    conf = locals()
    # Check if this phase is already done (if so, skip it)
    if skip(splits, save_folder, locals()):
        print("Skipping preparation, completed in previous run.")
        return

    # Additional checks to make sure the data folder contains VoxCeleb data
    if "," in data_folder:
        data_folder = [Path(dir) for dir in data_folder.replace(" ", "").split(",")]
    else:
        data_folder = [Path(data_folder)]

    print("Creating csv file for the Libri Dataset...")

    # Split data into 90% train and 10% validation (verification split)
    wav_lst_train, wav_lst_dev, utt2spk = _get_utt_split_lists(
        data_folder, split_ratio, num_utt, num_spk, utt_selected_ways
    )

    # Creating csv file for training data
    if "train" in splits:
        train_spks = prepare_csv(
            seg_dur, wav_lst_train, utt2spk, save_csv_train, random_segment=random_segment, amp_th=amp_th
        )
    print(f"Number of train speakers after amplitute filtering: {len(train_spks)}")

    if "dev" in splits:
        prepare_csv(seg_dur, wav_lst_dev, utt2spk, save_csv_dev, train_spks=train_spks, random_segment=random_segment, amp_th=amp_th)


    # Saving options (useful to skip this phase when already done)
    save_pkl(conf, str(save_opt))


# Used for verification split
def _get_utt_split_lists(
    data_folders, split_ratio, num_utt='ALL', num_spk='ALL', utt_selected_ways="spk-random"
):
    """
    Tot. number of speakers libri-360=921
    Splits the audio file list into train and dev.
    """
    train_lst = []
    dev_lst = []

    train_spks = set()
    dev_spks = set()

    print("Getting file list...")

    out_utt2spk = []
    for data_folder in data_folders:
        spk2utt = read_kaldi_format(data_folder / 'spk2utt')
        utt2spk = read_kaldi_format(data_folder / 'utt2spk')
        wavscp = read_kaldi_format(data_folder / 'wav.scp')
        out_utt2spk += utt2spk

        for spk, v in spk2utt.items():
            if 'list' in str(type(v)):
                utterances_of_speaker = [item for item in v]
                random.shuffle(utterances_of_speaker)
                split = int(0.01 * split_ratio[0] * len(utterances_of_speaker))
                train_ids = utterances_of_speaker[:split]
                dev_ids = utterances_of_speaker[split:]
            else:
                train_ids = [v]
                dev_ids = []
            assert len(train_ids) > 0
            if len(train_ids) > 0:
                train_spks.add(spk)
            if len(dev_ids) > 0:
                dev_spks.add(spk)

            train_snts = [wavscp[_id] for _id in train_ids]
            dev_snts = [wavscp[_id] for _id in dev_ids]

            train_lst.extend(train_snts)
            dev_lst.extend(dev_snts)

    print("Number of training samples:", len(train_lst))
    print("Number of development samples:", len(dev_lst))

    print("Number of training speakers:", len(train_spks))
    print("Number of development speakers:", len(dev_spks))

    return train_lst, dev_lst, out_utt2spk


def _get_chunks(seg_dur, audio_id, audio_duration):
    """
    Returns list of chunks
    """
    num_chunks = int(audio_duration / seg_dur)  # all in milliseconds
    chunk_lst = [
        audio_id + "_" + str(i * seg_dur) + "_" + str(i * seg_dur + seg_dur)
        for i in range(num_chunks)
    ]
    return chunk_lst


def prepare_csv(seg_dur, wav_lst, utt2spk, csv_file, train_spks=None, random_segment=False, amp_th=0):
    """
    Creates the csv file given a list of wav files.

    Arguments
    ---------
    wav_lst : list
        The list of wav files of a given data split.
    csv_file : str
        The path of the output csv file
    random_segment: bool
        Read random segments
    amp_th: float
        Threshold on the average amplitude on the chunk.
        If under this threshold, the chunk is discarded.

    Returns
    -------
    None
    """

    print(f"Creating csv lists in {csv_file}...")

    csv_output = [["ID", "duration", "wav", "start", "stop", "spk_id"]]

    spks = set()

    # For assigning unique ID to each chunk
    my_sep = "--"
    entry = []
    problematic_wavs = []
    # Processing all the wav files in the list
    for wav_file in tqdm(wav_lst, dynamic_ncols=True):
        # Getting sentence and speaker ids
        # TODO use utt2spk (but with the current impl, wav_file loses it's uniq id)
        try:
            temp = wav_file.split("/")[-1].split(".")[0]
            spk_id, utt_id = temp.split('_')
        except ValueError:
            print(f"Malformed path: {wav_file}")
            continue
        audio_id = my_sep.join([spk_id, utt_id.split(".")[0]])

        # if train_spks is given and spk_id is not in train_spks, skip
        if train_spks is not None and spk_id not in train_spks:
            continue

        # Reading the signal (to retrieve duration in seconds)
        try:
            audio_duration = sf.info(wav_file).duration
            #signal, fs = torchaudio.load(wav_file)
        except RuntimeError:
            problematic_wavs.append(wav_file)
            continue
        #signal = signal.squeeze(0)

        if random_segment:
            #audio_duration = signal.shape[0] / SAMPLERATE
            start_sample = 0
            #stop_sample = signal.shape[0]
            stop_sample = int(audio_duration * SAMPLERATE)

            # Composition of the csv_line
            csv_line = [
                audio_id,
                str(audio_duration),
                wav_file,
                start_sample,
                stop_sample,
                spk_id,
            ]
            entry.append(csv_line)
            spks.add(spk_id)
        else:
            #audio_duration = signal.shape[0] / SAMPLERATE
            signal, fs = torchaudio.load(wav_file)
            signal = signal.squeeze(0)

            uniq_chunks_list = _get_chunks(seg_dur, audio_id, audio_duration)
            for chunk in uniq_chunks_list:
                s, e = chunk.split("_")[-2:]
                start_sample = int(float(s) * SAMPLERATE)
                end_sample = int(float(e) * SAMPLERATE)

                #  Avoid chunks with very small energy
                mean_sig = torch.mean(np.abs(signal[start_sample:end_sample]))
                if mean_sig < amp_th:
                    continue

                # Composition of the csv_line
                csv_line = [
                    chunk,
                    str(audio_duration),
                    wav_file,
                    start_sample,
                    end_sample,
                    spk_id,
                ]
                entry.append(csv_line)
                spks.add(spk_id)

    print(f'Skipped {len(problematic_wavs)} invalid audios')
    csv_output = csv_output + entry

    # Writing the csv lines
    with open(csv_file, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        for line in csv_output:
            csv_writer.writerow(line)

    return spks

def skip(splits, save_folder, conf):
    """
    Detects if the voxceleb data_preparation has been already done.
    If the preparation has been done, we can skip it.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    # Checking csv files
    skip = True

    split_files = {
        "train": TRAIN_CSV,
        "dev": DEV_CSV,
    }
    for split in splits:
        if not Path(save_folder, split_files[split]).is_file():
            skip = False
    #  Checking saved options
    save_opt = save_folder / OPT_FILE
    if skip is True:
        if save_opt.is_file():
            opts_old = load_pkl(str(save_opt))

            if opts_old.popitem() == conf.popitem():
                skip = True
            else:
                skip = False
        else:
            skip = False

    return skip
