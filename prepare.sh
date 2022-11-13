#!/usr/bin/env bash

wget -O model.tar.gz https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_large_100_224/classification/5/default/1?tfjs-format=compressed
mkdir -p model
tar -xf model.tar.gz -C model
rm model.tar.gz

