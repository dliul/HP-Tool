#!/bin/bash

if [ ! -e ./config.properties ]; then
  cat >&2 <<-'EOF'
Error: Can't find configuration file in this folder.
Please download config.properties from our site and put it in this folder
EOF
  exit 1
fi
source ./config.properties

download(){
  echo "downloading $2"
  curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=$1" > /dev/null
  curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=$1" -o $2
  rm ./cookie
}

null_error(){
  cat >&2 <<-EOF
Error: The value of $1 is NULL, but it is required.
EOF
  exit 1
}

prepare_model(){
  # $1=inception_v3
  eval model="$""$1"
  if [ ${model} ]; then
    echo "Need Run $1: ${model}"
    if [ ${model} == "YES" ]; then
      eval model_name="$""$1_name"
      if [ ! ${model_name} ]; then
        null_error model_name
      fi
      eval model_dataset="$""$1_dataset"
      if [ ! ${model_dataset} ]; then
        null_error model_dataset
      fi

      eval model_url="$""$1_url"
      if [ ! -e ./models/${model_name} ]; then
        if [ ! ${model_url} ]; then
          null_error model_url
        fi
        download ${model_url} "./models/${model_name}"
      fi
      if [ ! -d "./dataset/${model_dataset}" ]; then
        dataset_url=${!model_dataset}
        if [ ! ${dataset_url} ]; then
          null_error "$1_dataset"
        fi
        download ${dataset_url} "./dataset/${model_dataset}.tar.gz"
        cd ./dataset
        tar -xzvf "${model_dataset}.tar.gz"
        rm "${model_dataset}.tar.gz"
        cd ..
      fi
      if [ ! -e ./dataset/"${model_dataset}.gtruth" ]; then
        eval dataset_gtruth="$""${model_dataset}_gtruth"
        if [ ! ${dataset_gtruth} ]; then
          null_error "${model_dataset}_gtruth"
        fi
        download ${dataset_gtruth} ./dataset/"${model_dataset}.gtruth"
      fi
    fi
  fi
}

[ ! -d "./models/" ] && (mkdir ./models/)
[ ! -d "./dataset/" ] && (mkdir ./dataset/)

prepare_model inception_v3
prepare_model resnet50_v1
prepare_model mobilenet_v3