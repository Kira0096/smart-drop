import time
import torch
from resnet_for_vis import ResNet18
from keras.datasets import cifar10
import numpy as np

#region Data Preprocess
t0 = time.time()
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
(x_train, y_train), (x_test, y_test) = (np.array(x_train), np.array(y_train)), (np.array(x_test), np.array(y_test))
# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
# If subtract pixel mean is enabled
# normalize = transforms.Normalize(mean=[x / 255.0 for x in [123.675, 116.28, 103.53]],
#                                  std=[x / 255.0 for x in [58.395, 57.12, 57.375]])
# import ipdb; ipdb.set_trace()
x_train_mean = np.array([x / 255.0 for x in [123.675, 116.28, 103.53]])
x_train_std = np.array([x / 255.0 for x in [58.395, 57.12, 57.375]])
x_train =  (x_train - x_train_mean) / x_train_std
x_test =  (x_test - x_train_mean) / x_train_std

x_train = np.rollaxis(x_train,3,1)
x_test = np.rollaxis(x_test,3,1)

num_classes = 10
# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# define model
model = ResNet18(num_classes=num_classes)
checkpoint_name = 'Test'
checkpoint_name = 'adadrop_cifar10'
checkpoint_name = 'dropblock_cifar10'
checkpoint_path = '/unsullied/sharefs/ouyangzhihao/DataRoot/Exp/Tsinghua/DropOut/checkpoints/checkpoints/' + checkpoint_name + '.pt'
model.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage.cuda(0)))

#endregion

batchsize = 512

img_tensor = torch.from_numpy(x_test).float()
label_tensor = torch.from_numpy(y_test.flatten())

model.to('cuda')
img_tensor = img_tensor.to('cuda')
label_tensor = label_tensor.to('cuda')
model.eval()
final_res = []

from progressbar import *
pbar = ProgressBar()
print('Start training~')

# for i in pbar(range(0,len(x_train),batchsize)):
#     # print(i)
#     batch_data = img_tensor[i:i+batchsize]
#     res = model(batch_data)
#     final_res.append(res)
#     if(i > 2): break

i = 0
batch_data = img_tensor[i:i+batchsize]
res = model(batch_data)
final_res.append(res)

final_res = np.array(final_res)
# import ipdb; ipdb.set_trace()
print('final_res shape', np.shape(final_res))
print('final_res[0] shape',np.shape(final_res[0]))

np.save('cutout_cifar10', final_res)

predict_res = res[0]
# import ipdb; ipdb.set_trace()
predict_res = torch.argmax(predict_res, dim=1)
correct = (predict_res == label_tensor[:batchsize]).sum().item()

accuracy = correct / batchsize

# accuracy = (torch.argmax(predict_res, dim=1).numpy() == y_train[:batchsize].flatten()).sum() / batchsize


# results = model.conv1.forward(img_tensor)
t1 = time.time()
print('Test Accuracy:', accuracy)
print('Total time:', t1 - t0)