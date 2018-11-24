echo $*
GLOG_vmodule=MemcachedClient=-1 srun --mpi=pmi2 -p $1 -n$2 --gres=gpu:$2 --ntasks-per-node=$2 python resnet-cifar10.py  --device 0 $3 $4 $5 $6 $7
