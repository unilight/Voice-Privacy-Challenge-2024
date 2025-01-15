

import argparse
import os
import librosa
import torch
from tqdm import tqdm


from transformers import WhisperProcessor, WhisperForConditionalGeneration
import jiwer
import pyopenjtalk

def load_asr_model(asr_engine, device):
    """Load model"""
    if asr_engine == "whisperx":
        import whisperx
        device = "cuda" if torch.cuda.is_available() else "cpu"
        models = whisperx.load_model("large-v2", device, compute_type="float16", language="ja")
    
    elif asr_engine == "whisper":
        
        ASR_PRETRAINED_MODEL = "vumichien/whisper-large-v2-jp"

        print(f"[INFO]: Load the pre-trained ASR by {ASR_PRETRAINED_MODEL}.")
        processor = WhisperProcessor.from_pretrained(ASR_PRETRAINED_MODEL)
        model = WhisperForConditionalGeneration.from_pretrained(ASR_PRETRAINED_MODEL).to(
            device
        )
        models = {"model": model, "processor": processor}

    elif asr_engine == "nue_asr":
        import nue_asr
        model = nue_asr.load_model("rinna/nue-asr")
        tokenizer = nue_asr.load_tokenizer("rinna/nue-asr")
        models = {"model": model, "tokenizer": tokenizer}
    return models

def normalize_sentence(sentence):
    """Normalize sentence"""
    # Convert all characters to upper.
    sentence = sentence.upper()
    # Delete punctuations.
    sentence = jiwer.RemovePunctuation()(sentence)
    sentence = pyopenjtalk.g2p(sentence, kana=True)

    return sentence


def transcribe(model, asr_engine, device, wav_path):
    """Calculate score on one single waveform"""
    # preparation

    if asr_engine == "whisperx":
        import whisperx
        audio = whisperx.load_audio(wav_path)
        transcription = model.transcribe(audio, batch_size=1)["segments"][0]["text"]
    elif asr_engine == "whisper":
        # load waveform
        wav, _ = librosa.load(wav_path, sr=16000)
        inputs = model["processor"](
            wav, sampling_rate=16000, return_tensors="pt"
        ).input_features
        inputs = inputs.to(device)

        # forward
        predicted_ids = model["model"].generate(inputs)
        transcription = model["processor"].batch_decode(
            predicted_ids, skip_special_tokens=True
        )
    elif asr_engine == "nue_asr":
        import nue_asr
        transcription = nue_asr.transcribe(model["model"], model["tokenizer"], wav_path).text
    return transcription


def calculate_measures(groundtruth, transcription):
    """Calculate character/word measures (hits, subs, inserts, deletes) for one given sentence"""
    groundtruth = normalize_sentence(groundtruth)
    transcription = normalize_sentence(transcription)

    c_result = jiwer.cer(groundtruth, transcription, return_dict=True)
    w_result = jiwer.compute_measures(groundtruth, transcription)

    return c_result, w_result, groundtruth, transcription

def get_basename(path):
    return os.path.splitext(os.path.split(path)[-1])[0]

def _calculate_asr_score(model, asr_engine, device, wav_paths, gt_texts):
    keys = ["hits", "substitutions", "deletions", "insertions"]
    ers = {}
    c_results = {k: 0 for k in keys}
    w_results = {k: 0 for k in keys}

    for _id, wav_path in tqdm(wav_paths.items()):
        groundtruth = gt_texts[_id]  # get rid of the first character "E"

        # trascribe
        transcription = transcribe(model, asr_engine, device, wav_path)
        transcription = "".join(str(i) for i in transcription)
        transcription = transcription.replace(" ", "")

        # error calculation
        c_result, w_result, norm_groundtruth, norm_transcription = calculate_measures(
            groundtruth, transcription
        )

        ers[_id] = [
            c_result["cer"] * 100.0,
            w_result["wer"] * 100.0,
            norm_transcription,
            norm_groundtruth,
        ]

        for k in keys:
            c_results[k] += c_result[k]
            w_results[k] += w_result[k]

    # calculate over whole set
    def er(r):
        return (
            float(r["substitutions"] + r["deletions"] + r["insertions"])
            / float(r["substitutions"] + r["deletions"] + r["hits"])
            * 100.0
        )

    cer = er(c_results)
    wer = er(w_results)

    return ers, cer, wer

def read_file(path):
    with open(path, "r") as f:
        lines = f.read().splitlines()
    ret = {}
    for line in lines:
        _id, content = line.split(" ")
        ret[_id] = content
    return ret

def get_parser():
    parser = argparse.ArgumentParser(description="objective evaluation script.")
    parser.add_argument("--wavscp", required=True, type=str, help="wav scp")
    parser.add_argument("--text", required=True, type=str, help="text")
    parser.add_argument("--out", required=True, type=str, help="out")
    parser.add_argument("--asr_engine", type=str, required=True, choices=["whisper", "whisperx", "nue_asr"])
    return parser


def main():
    args = get_parser().parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Load wav.scp")
    wav_paths = read_file(args.wavscp)

    print("Load ground text")
    gt_texts = read_file(args.text)

    print("Calculating ASR-based score...")
    # load ASR model
    asr_model = load_asr_model(args.asr_engine, device)

    # calculate error rates
    ers, cer, wer = _calculate_asr_score(
        asr_model, args.asr_engine, device, wav_paths, gt_texts
    )

    print(f"CER: {cer}")

    with open(args.out, "w") as f:
        for _id, result in ers.items():
            cer = result[0]
            gt_text = result[2]
            syn_text = result[3]

            f.write(f"{_id},{cer},{gt_text},{syn_text}\n")

    

if __name__ == "__main__":
    main()