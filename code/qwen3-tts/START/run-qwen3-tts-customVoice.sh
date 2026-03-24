#!/bin/bash

tmp_log=$(mktemp)


# lets use a working model
export MODEL_NAME=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice

export INFERENCE_NAME=Qwen3-TTS-$(whoami)-$(uuidgen | cut -d '-' -f 1)

export TENSOR_PARALLEL_SIZE=1

export RUNTIME=vllm-omni-0.14.0


flexai inference serve $INFERENCE_NAME \
 --affinity "cluster=k8s-training-smc-001" \
 --runtime ${RUNTIME} \
 --accels ${TENSOR_PARALLEL_SIZE} \
  -- $MODEL_NAME --stage-configs-path /workspace/vllm-omni/vllm_omni/model_executor/stage_configs/qwen3_tts.yaml --omni --trust-remote-code --enforce-eager \
2>&1 | tee "${tmp_log}"

export INFERENCE_API_KEY_TTS=$(grep 'API Key' ${tmp_log} | cut -d ':'  -f 2 | tr -d ' ')

sleep 10


# wait for the inference to be ready
STATUS="enqueued"

while [ ${STATUS} != "running" ]; do
    STATUS=$(flexai inference inspect $INFERENCE_NAME -j | jq -r .runtime.status)
    echo "${STATUS}"
    sleep 10
done


export INFERENCE_URL_TTS=$(flexai inference inspect $INFERENCE_NAME -j | jq .config.endpointUrl -r)
