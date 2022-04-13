#!/bin/bash

MODEL_URL="https://storage.googleapis.com/audioset/vggish_model.ckpt"
PCA_URL="https://storage.googleapis.com/audioset/vggish_pca_params.npz"

MODEL_DIR=openmic/vggish/_model
MODEL_PARAMS=${MODEL_DIR}/vggish_model.ckpt
PCA_PARAMS=${MODEL_DIR}/vggish_pca_params.npz

mkdir -p ${MODEL_DIR}

if [ -e ${MODEL_PARAMS} ]
then
    echo "Model parameters exist; skipping."
else
    wget ${MODEL_URL} -O ${MODEL_PARAMS}
fi

if [ -e ${PCA_PARAMS} ]
then
    echo "Model parameters exist; skipping."
else
    wget ${PCA_URL} -O ${PCA_PARAMS}
fi
