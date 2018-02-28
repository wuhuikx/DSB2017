export PYTHONPATH=/home/sgf/tianchi/extended-caffe/python:$PYTHONPATH
export LD_LIBRARY_PATH=/home/sgf/tianchi/extended-caffe/external/mkldnn/install/lib:$LD_LIBRARY_PATH

unset OMP_NUM_THREADS
export OMP_NUM_THREADS=67
export MKL_NUM_THREADS=67

unset MKL_THREADING_LAYER

export KMP_AFFINITY=explicit,granularity=fine,proclist=[0-66]

python train.py  # if train, uncomment this line
#python test.py  # if test, uncomment this line
