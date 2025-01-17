import os
import soundfile as sf
from tqdm import tqdm

ROOT_DIR = "/data/group1/z44476r/Corpora/JTubeSpeech-ASV"
WAV_DIR = os.path.join(ROOT_DIR, "wav")

os.makedirs(WAV_DIR, exist_ok=True)

# train files look like train/ACA/ACA_3YjThnkv.mp3
for spk in tqdm(os.listdir(os.path.join(ROOT_DIR, "train"))):
    for mp3_path in os.listdir(os.path.join(ROOT_DIR, "train", spk)):
        full_mp3_path = os.path.join(ROOT_DIR, "train", spk, mp3_path)
        wav_path = os.path.join(ROOT_DIR, "wav", "train", spk, mp3_path.replace(".mp3", ".wav"))
        os.makedirs(os.path.join(ROOT_DIR, "wav", "train", spk), exist_ok=True)

        if os.path.exists(wav_path):
            continue
        
        data, samplerate = sf.read(full_mp3_path)
        sf.write(wav_path, data, samplerate)

for _set in ["query", "enr"]: 
    for spk in tqdm(os.listdir(os.path.join(ROOT_DIR, "test", _set))):
        for session in os.listdir(os.path.join(ROOT_DIR, "test", _set, spk)):
            for mp3_path in os.listdir(os.path.join(ROOT_DIR, "test", _set, spk, session)):
                full_mp3_path = os.path.join(ROOT_DIR, "test", _set, spk, session, mp3_path)
                wav_path = os.path.join(ROOT_DIR, "wav", "test", _set, spk, session, mp3_path.replace(".mp3", ".wav"))
                os.makedirs(os.path.join(ROOT_DIR, "wav", "test", _set, spk, session), exist_ok=True)

                if os.path.exists(wav_path):
                    continue
                
                data, samplerate = sf.read(full_mp3_path)
                sf.write(wav_path, data, samplerate)

