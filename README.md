# ResNet18 on CIFAR-10


## Requirements

```bash
pip install -r requirements.txt
```
Change the root of your cifar10/cifar100 in config

## Usage

Run the example on the CPU:

```bash
python resnet-cifar10.py -n EXPNAME -c config.yml
```

Run the example on the GPU (device 0):

```bash
python resnet-cifar10.py -n EXPNAME -c config.yml --device 0
```

Draw Masks:

```bash
python save_mask.py CHECKPOINT FILELIST
```
