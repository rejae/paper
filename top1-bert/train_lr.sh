#!/bin/bash
export PYTHONIOENCODING=UTF-8

#cd jss_util
#python3 download.py $*
#if [ $? -ne 0 ]; then
#  exit 1
#fi

cd lr
python3 dataPreperation.py $*
if [ $? -ne 0 ]; then
  exit 1
fi

java -jar -Xmx5500m -Xms5500m train.jar $*
if [ $? -ne 0 ]; then
  exit 1
fi

python3 dataReorganize.py $*
if [ $? -ne 0 ]; then
  exit 1
fi

cd jss_util
python3 upload.py lr $*

cd ../..
python3 rmLrModelFile.py $*
if [ $? -ne 0 ]; then
  exit 1
fi
