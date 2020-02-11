#!/bin/bash


source config.properties
current_path=$(cd "$(dirname $0)";pwd)
source ${current_path}/tx2.properties

null_error(){
  cat >&2 <<-EOF
Error: The value of $1 is NULL, but it is required.
EOF
  exit 1
}

no_exist_error(){
  cat >&2 <<-EOF
Error: The folder or file $1 does not exist, but it is required.
Are you sure you have execused download.sh or prepared required items manually?
EOF

  exit 1
}

run_models(){
  eval model="$""$1"
  if [ ${model} ]; then
    if [ ${model} == "YES" ]; then
      echo "Runing $1 on TX2"
      eval model_name="$""$1_name"
      if [ ! ${model_name} ]; then
        null_error model_name
      fi
      modelpath="${current_path}/../../models/${model_name}"
      eval model_dataset="$""$1_dataset"
      if [ ! ${model_dataset} ]; then
        null_error model_dataset
      fi
      datasetpath="${current_path}/../../dataset/${model_dataset}/"
      groundtruth="${current_path}/../../dataset/${model_dataset}.gtruth"
      if [ ${tensorflow_GPU} == "YES" ]; then
        python3 ${current_path}/classification.py --modelpath=${modelpath} --dataset=${datasetpath} --normalization=1 --precision=GPU --labelpath=${groundtruth} --batchsize=16 --loadnumber=512 --resultname=${current_path}/rr.json
      fi
      if [ ${tensorflow_CPU} == "YES" ]; then
        python3 ${current_path}/classification.py --modelpath=${modelpath} --dataset=${datasetpath} --normalization=1 --precision=CPU --labelpath=${groundtruth} --batchsize=16 --loadnumber=512 --resultname=${current_path}/rr.json
      fi
      if [ ${tensorRT_INT8} == "YES" ]; then
        len=${#modelpath}
        model_int8="${modelpath:0:$len-3}_INT8.pb"
        [ ! -e ${model_int8} ] && (python3 ${current_path}/convert_TF_RT.py --modelpath=${modelpath} --calib_img=${datasetpath} --normalization=1 --precision=INT8)
        python3 ${current_path}/classification.py --modelpath=${model_int8} --dataset=${datasetpath} --normalization=1 --precision=INT8 --labelpath=${groundtruth} --batchsize=16 --loadnumber=512 --resultname=${current_path}/rr.json
      fi
      if [ ${tensorRT_FP16} == "YES" ]; then
        len=${#modelpath}
        model_fp16="${modelpath:0:$len-3}_FP16.pb"
        [ ! -e ${model_fp16} ] && (python3 ${current_path}/convert_TF_RT.py --modelpath=${modelpath} --calib_img=${datasetpath} --normalization=1 --precision=FP16)
        python3 ${current_path}/classification.py --modelpath=${model_fp16} --dataset=${datasetpath} --normalization=1 --precision=FP16 --labelpath=${groundtruth} --batchsize=16 --loadnumber=512 --resultname=${current_path}/rr.json
      fi
      if [ ${tensorRT_FP32} == "YES" ]; then
        len=${#modelpath}
        model_fp32="${modelpath:0:$len-3}_FP32.pb"
        [ ! -e ${model_fp32} ] && (python3 ${current_path}/convert_TF_RT.py --modelpath=${modelpath} --calib_img=${datasetpath} --normalization=1 --precision=FP32)
        python3 ${current_path}/classification.py --modelpath=${model_fp32} --dataset=${datasetpath} --normalization=1 --precision=FP32 --labelpath=${groundtruth} --batchsize=16 --loadnumber=512 --resultname=${current_path}/rr.json
      fi
    fi
  fi
}

run_models resnet50_v1