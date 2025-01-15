#!/bin/bash

# Copyright 2025 Wen-Chin Huang (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

source env.sh

# basic settings
stage=1       # stage to start
stop_stage=100 # stage to stop

# dataset setting
jvs_db_root=/data/group1/z44476r/Corpora/jvs_ver1
jtubespeech_db_root=/data/group1/z44476r/Corpora/JTubeSpeech-ASV

# configs (in VPC, pre = no asv fine-tuning; post = with fine-tuning)
anon_config=configs/anon_mcadams_jvs_jtube.yaml
oa_config=configs/eval_pre_jtube.yaml
aa_config=configs/eval_post_jtube.yaml

anon_data_suffix=_mcadams

force_compute=
# force_compute='--force_compute True'

# shellcheck disable=SC1091
. ./parse_options.sh || exit 1;

set -euo pipefail

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    log "stage -1: Model download and data preparation. This might take a while."

    ./01b_download_evaluation_models.sh

    python replace_db_root.py --wavscp data/jvs_train/wav.scp --old_db_root "/data/group1/z44476r/Corpora/jvs_ver1" --new_db_root "${jvs_db_root}"
    python replace_db_root.py --wavscp data/jtubespeech-asv_train/wav.scp --old_db_root "/data/group1/z44476r/Corpora/JTubeSpeech-ASV" --new_db_root "${jtubespeech_db_root_db_root}"
    python replace_db_root.py --wavscp data/jtubespeech-asv_test_2s_enrolls/wav.scp --old_db_root "/data/group1/z44476r/Corpora/JTubeSpeech-ASV" --new_db_root "${jtubespeech_db_root_db_root}"
    python replace_db_root.py --wavscp data/jtubespeech-asv_test_2s_trials/wav.scp --old_db_root "/data/group1/z44476r/Corpora/JTubeSpeech-ASV" --new_db_root "${jtubespeech_db_root_db_root}"
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Anonymization using methods supported in VPC'24 official toolkit"

    python run_anonymization.py --config ${anon_config} ${force_compute}
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Evaluation - automatic speech recognition"

    python run_ja_asr.py \
        --wavscp "data/jvs_train${anon_data_suffix}/wav.scp" \
        --text "data/jvs_train${anon_data_suffix}/text" \
        --out "data/jvs_train${anon_data_suffix}/asr_results.csv" \
        --asr_engine nue_asr
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Evaluation - Automatic speech verification (OA)"

    python run_evaluation.py \
        --config "${oa_config}" \
        --overwrite "{\"anon_data_suffix\": \"$anon_data_suffix\"}" ${force_compute}
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: Evaluation - Automatic speech verification (AA)"

    python run_evaluation.py \
        --config "${aa_config}" \
        --overwrite "{\"anon_data_suffix\": \"$anon_data_suffix\"}" ${force_compute}
fi
