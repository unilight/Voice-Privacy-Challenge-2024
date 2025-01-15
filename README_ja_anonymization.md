# Evaluating speaker anonymization systems in Japanese

Jan 2025 by Wen-Chin Huang

## Introduction

I extended the VoicePrivacy Challenge (VPC) 2024 toolkit so that it supports evaluation in Japanese.

**About OA and AA**: The ASV evaluation is usually divided into OA and AA. To understand the difference, I recommend reading the VPC'20 summary paper (cf. Sec. 2.3.1 https://arxiv.org/pdf/2109.00648). In practice, for OA, we just use the pre-trained ASV model. For AA, we need to fine-tune the ASV model using anonymized data. This will help understand what we do below.

## Usage

### Installation

First, follow the official instruction:

```
./00_install.sh
```

Then, we need to install nue-asr (https://huggingface.co/rinna/nue-asr):

```
source env.sh
pip install git+https://github.com/rinnakk/nue-asr.git
```

### Dataset downloading

We need two datasets. Please download and extract them somewhere in your local machine.

#### JVS
Link: https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus?authuser=0  
The structure should look like:
```
z44476r@flow-cx02:/data/group1/z44476r/Corpora/jvs_ver1$ ls
jvs001/  jvs002/  ...
```

#### JTubeSpeech-ASV
Link: https://sites.google.com/site/shinnosuketakamichi/research-topics/jtubespeech-asv_corpus  
The structure should look like:
```
z44476r@flow-cx02:/data/group1/z44476r/Corpora/JTubeSpeech-ASV$ ls
meta/  test/  train/  wav/ ...
```

### Data format

VPC'24 puts all the data, including the original and anonymized data, in the `data` directory. In each sub-directory, we need to follow the Kaldi format (very troublesome, I know). I've put the necessary directories for the above datasets in the `data` directory, including:

- `jvs_train`
- `jtubespeech-asv_train`: this is used to fine-tune the ASV model.
- `jtubespeech-asv_test_2s_enrolls`
- `jtubespeech-asv_test_2s_trials`

The structure looks like this:
```
├── jtubespeech-asv_test_2s_enrolls
│   ├── enrolls
│   ├── spk2gender
│   ├── spk2utt
│   ├── utt2spk
│   └── wav.scp
├── jtubespeech-asv_test_2s_trials
│   ├── spk2gender
│   ├── spk2utt
│   ├── trials
│   ├── utt2spk
│   └── wav.scp
├── jtubespeech-asv_train
│   ├── spk2utt
│   ├── utt2spk
│   └── wav.scp
├── jvs_train
│   ├── segments
│   ├── spk2gender
│   ├── spk2utt
│   ├── text
│   ├── utt2spk
│   └── wav.scp
└── out
```

You need to anonymize each directory, and make new directories with the name like `jvs_train_<suffix>`, etc. So if your anonymization method's name is `mhubert`, then the anonymized folder looks like `jvs_train_mhubert`. The anonymized wav files can be put at any location in your local machine, but you need to prepare the `jvs_train_mhubert/wav.scp` file, with the first column be the ID, the second column be the anonymized wav path:

```
jvs001_nonparallel_BASIC5000_0025 /data/group1/z44476r/Experiments/Voice-Privacy-Challenge-2024/data/jvs_train_mcadams/wav/jvs001_nonparallel_BASIC5000_0025.wav
...
```

### Execution

The following shell script file covers every stage:

```
./run_ja.sh 
```

You can specify the stage like this:

```
./run_ja.sh --stage 1 --stop_stage 1
```

You can just execute the script as it is to get to know how it works (but it'll take several hours). Afterwards, if you have prepared your anonymized data, and let's say the method's name is `mhubert`, then you can run the following:

```
./run_ja.sh --stage 1 --anon_data_suffix mhubert
```


#### Stage -1: Download the pre-trained ASV model, and do data preparation.

First, the script downloads the pre-trained ASV model.

Then we do data preparation. Essentially, what we do is to modify the `wav.scp` files: the wav paths in the scp files are absolute files, so I wrote a script to modify the dataset root path. Let's say you put your JTubeSpeech-ASV dataset in `/foo/bar/JTubeSpeech-ASV`. Then, for example, the first line in `data/jtubespeech-asv_train/wav.scp` will be changed from 

```
AAA_iPVn3t8I /home/z44476r/data/Corpora/JTubeSpeech-ASV/wav/train/AAA/AAA_iPVn3t8I.wav
```

to this:

```
AAA_iPVn3t8I /foo/bar/JTubeSpeech-ASV/wav/train/AAA/AAA_iPVn3t8I.wav
```

#### Stage 0: Anonymization using methods supported in VPC'24 official toolkit

This step is not necessary, but it just serves as a sanity check. We use the `mcadams` method for example. After this is done, you will see the following directories: `data/jvs_train_mcadams`, `data/jtubespeech-asv_train_mcadams`, etc.

#### Stage 1: Automatic speech evaluation

Run ASR with a Japanese ASR model. Here we use the NUE ASR model provided here: https://huggingface.co/rinna/nue-asr. This might take a couple of hours.

#### Stage 2: Automatic speech verification (OA)

Run ASV in the OA scenario.

#### Stage 3: Automatic speech verification (AA)

Run ASV in the AA scenario. This might take several hours.