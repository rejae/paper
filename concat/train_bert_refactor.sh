#!/bin/bash
export PYTHONIOENCODING=UTF-8

#cd jss_util
#python3 download.py $*
#if [ $? -ne 0 ]; then
#  exit 1
#fi
#echo "download.py finish"

cd bert
python3 dataPreperation_refactor.py $*
if [ $? -ne 0 ]; then
  exit 1
fi
echo "dataPreperation_refactor.py finish"





