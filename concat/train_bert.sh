#!/bin/bash
export PYTHONIOENCODING=UTF-8

#cd jss_util
#python3 download.py $*
#if [ $? -ne 0 ]; then
#  exit 1
#fi
#echo "download.py finish"

cd bert
python3 dataPreperation.py $*
if [ $? -ne 0 ]; then
  exit 1
fi
echo "dataPreperation.py finish"

python3 bertTrainAndTest.py $*
if [ $? -ne 0 ]; then
  exit 1
fi
echo "bertTrainAndTest.py finish"







:<<EOF
python3 modelOptimizer.py $*
if [ $? -ne 0 ]; then
  exit 1
fi
echo "modelOptimizer.py finish"

python3 dataReorganize.py $*
if [ $? -ne 0 ]; then
  exit 1
fi
echo "dataReorganize.py finish"

cd ../jss_util
python3 upload.py bert $*
if [ $? -ne 0 ]; then
  exit 1
fi
echo "upload.py finish"

cd ..
python3 rmModelFile.py $*
if [ $? -ne 0 ]; then
  exit 1
fi
echo "rmModelFile.py"
EOF

