import time
import torch
from pathlib import *
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# font = {'family' : 'Times Roman',}
#         # 'weight' : 'bold',
#         # 'size'   : 22}
# matplotlib.rc('font', **font)
# import ipdb; ipdb.set_trace()

plt.figure(figsize=(15,10))


# plt.rcParams["font.size"] = 22
#region Data Preprocess
t0 = time.time()
num_classes = 10
# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#endregion
exp_name = 'Test'
batchsize = 512

indexes = [1,2,3,4]
# import ipdb; ipdb.set_trace()
##cutout_cifar10 ;  adadrop_cifar10; dropblock_cifar10.npy
feature_maps1 = np.load('dropblock_cifar10' + '.npy')[0]
feature_maps2 = np.load('adadrop_cifar10' + '.npy')[0]


for idx in indexes:
    predict_res_1 = feature_maps1[idx].flatten()
    predict_res_2 = feature_maps2[idx].flatten()
    plt.subplot(2,2,idx)
    # import ipdb; ipdb.set_trace()
    origin_data = plt.hist(predict_res_1,bins=50,range=(0.1,1),color='b', label='dropblock')
    plt.hist(predict_res_2,bins=50,range=(0.1,1),color='r',alpha=0.5, label='adadrop')
    plt.legend(loc=0,ncol=2)#图例及位置： 1右上角，2 左上角 loc函数可不写 0为最优 ncol为标签有几列
    plt.savefig('./picture/dropblock_adadrop_layer%d.pdf' % idx)

pic_path = Path('./picture/')
pic_path.mkdir(parents=True, exist_ok=True)
plt.savefig('./picture/dropblock_adadrop_1-4layer.pdf')
plt.close()

# results = model.conv1.forward(img_tensor)
t1 = time.time()
# print('Test Accuracy:', accuracy)
print('Total time:', t1 - t0)