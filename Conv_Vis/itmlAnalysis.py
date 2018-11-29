import time
import torch
import cv2
import io
import requests
from PIL import Image
from torchvision import models, transforms
from PIL import Image
from torch.autograd import Variable
from torch.nn import functional as F
from resnet_for_vis import ResNet18
from keras.datasets import cifar10
from torchvision import models, transforms
import numpy as np
import os
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
# x_test -= x_train_mean
x_train = np.rollaxis(x_train,3,1)
# x_test = np.rollaxis(x_test,3,1)

num_classes = 10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# define model
model = ResNet18(num_classes=num_classes)
model.load_state_dict(torch.load('/home/ouyangzhihao/sss/Exp/HTB/smart-drop/checkpoints/' + 'Test' + '.pt', map_location=lambda storage, loc: storage.cuda(0)))


batchsize = 128
part_data = x_train[:batchsize]
img_tensor = torch.from_numpy(part_data).float()

model.eval()

all_res = model(img_tensor)

# 该数组记录中间层结果
features_blobs = []
# 该函数由register_forward_hook调用，类似于event handler，当resnet前向传播时记录所需中间层结果
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())
# 需要输出的中间层名称，名称为resnet_for_vis的__init__函数中声明的。
finalconv_name = 'layer4'
model._modules.get(finalconv_name).register_forward_hook(hook_feature)
params = list(model.parameters())
# 记录layer4与之后层的权值矩阵，用于计算下一层feature map（由returnCAM函数实现），从而生成热力图。
weight_softmax = np.squeeze(params[-2].data.numpy())

# 函数输入：feature_conv是中间层（feature map），weight_softmax是中间层与下一层的权值矩阵，class_idx是下一层的类别向量。
# 功能：计算下一层的feature map，用于生成热力图（激活值高的温度高，表现在图像中就是红色）。
# 返回值：下一层的feature map
def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (32, 32)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Resize((32,32)),
   transforms.ToTensor(),
   normalize
])

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# data_idx是x_train的下标，用于打印想要查看的图片及其热力图。
data_idx = [0,1,2,3,4,5,6,7]
save_dir = os.path.join(os.getcwd(), 'pic')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
for i in range(len(data_idx)):
    img_pil = Image.fromarray(x_train[data_idx[i]], 'RGB')  # 输入格式应为32X32X3的numpy数组
    pic_name = './pic/oriPic{}'.format(i)
    img_pil.save(pic_name+'.jpg')
    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))
    res = model(img_variable)

    h_x = F.softmax(res, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()

    # generate class activation mapping for the top1 prediction

    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])
    img = cv2.imread(pic_name+'.jpg')
    # import ipdb;ipdb.set_trace()
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5

    label = classes[y_train[i][0]]
    cv2.imwrite(pic_name+label+'.jpg', result)