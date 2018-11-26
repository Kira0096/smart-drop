#echo $*
#GLOG_vmodule=MemcachedClient=-1 srun --mpi=pmi2 -p $1 -n$2 --gres=gpu:$2 --ntasks-per-node=$2 python resnet-cifar10.py  --device 0 $3 $4 $5 $6 $7
rlaunch --cpu=2 --gpu=1 --memory=4096 --preemptible=yes -- python3 resnet-cifar10.py -c config.yml