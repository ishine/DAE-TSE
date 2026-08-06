#!/bin/bash

. ./path.sh || exit 1

set -euo pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=1
stop_stage=2

nj=1

# Data preparation (WeSep / LibriMix meta)
data=data
fs=16k
min_max=min
noise_type=clean
Libri2Mix_dir=/path/to/Libri2Mix
mix_data_path="${Libri2Mix_dir}/wav${fs}/${min_max}"

# Inference
exp_dir=
checkpoint=checkpoint_150.pt
save_results=true
infer_tag=
test_text_enroll=data/text_cue/testset/kw-4_seed-42.jsonl

. scripts/kaldi/parse_options.sh || exit 1

# Recompute after parse_options so --Libri2Mix_dir takes effect
mix_data_path="${Libri2Mix_dir}/wav${fs}/${min_max}"

# Ctrl-C
function onCtrlC () {
    echo ""
    echo "[INFO] Ctrl+C is captured. Clear related processes."

    ps -aux | grep "python wesep/bin/infer.py" | grep "${config:-}" | grep -v grep | awk '{print $2}' | xargs kill -9
}

# Stage 1: prepare LibriMix test wav.scp (no enrollment needed for DAE-TSE)
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  log "Prepare LibriMix datasets ..."
  ./local/prepare_data.sh --mix_data_path ${mix_data_path} \
    --data ${data} \
    --noise_type ${noise_type} \
    --stage 1 \
    --stop-stage 1
fi

test_data=${data}/${noise_type}/test
echo "test_data: ${test_data}"

# Stage 2: inference (SI-SNR / SI-SNRi)
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  config=${exp_dir}/config.yaml
  ckpt=${exp_dir}/models/${checkpoint}
  log "checkpoint: ${ckpt}"
  log "Start inferencing ..."

  if [[ "${nj}" == "1" ]]; then
    python wesep/bin/infer.py --config $config \
      --fs ${fs} \
      --gpus 0 \
      --exp_dir ${exp_dir} \
      --data_type raw \
      --test_data ${test_data}/wav.scp \
      --save_wav ${save_results} \
      ${ckpt:+--checkpoint $ckpt} \
      ${infer_tag:+--infer_tag $infer_tag} \
      ${test_text_enroll:+--test_text_enroll $test_text_enroll}
  else
    trap 'onCtrlC' INT

    gpus=`seq 0 $(($nj - 1))`
    for i in ${gpus}; do
      python wesep/bin/infer.py --config $config \
        --fs ${fs} \
        --gpus ${i} \
        --exp_dir ${exp_dir} \
        --data_type raw \
        --test_data ${test_data}/split${nj}/${i}/wav.scp \
        --save_wav ${save_results} \
        ${ckpt:+--checkpoint $ckpt} \
        ${infer_tag:+--infer_tag $infer_tag} \
        ${test_text_enroll:+--test_text_enroll $test_text_enroll} &
    done
    wait
  fi

  if [ ${infer_tag} ]; then
    cat ${exp_dir}/inference/${infer_tag}/jsonl_split/infer.*.jsonl > ${exp_dir}/inference/${infer_tag}/infer.jsonl
    echo "Merged jsonl files to ${exp_dir}/inference/${infer_tag}/infer.jsonl"
    python local/merge_jsonl.py --jsonl_or_dir ${exp_dir}/inference/${infer_tag}/infer.jsonl
  fi
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
