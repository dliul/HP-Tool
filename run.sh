#!/bin/bash

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

usage(){
  echo "usge meassage"
}

if [ ! -d "./models/" ]; then
  no_exist_error ./models/
fi

if [ ! -d "./dataset/" ]; then
  no_exist_error ./dataset/
fi

if [ ! -d "./scripts/" ]; then
  no_exist_error ./scripts/
fi

case $1 in

  "tx2")
    echo "Runing Nvidia TX2"
    bash ./scripts/tx2/tx2.sh
    ;;
  "tpu")
    echo "Runing Edge TPU"
    ;;
  "ncs")
    echo "Runing Intel NCS2"
    ;;
  "all")
    echo "Runing all platform"
    ;;
  *)
    usage
    ;;
esac