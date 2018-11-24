import time
import configargparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models.resnet import BasicBlock, ResNet
from resnet import ResNet18
from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Events
from ignite.metrics import Accuracy
from ignite.metrics import RunningAverage
from ignite.contrib.handlers import ProgressBar

from dropblock.dropblock import DropBlock2D
from dropblock.scheduler import LinearScheduler


from torch.optim.lr_scheduler import MultiStepLR

results = []


class ResNetCustom(ResNet):

    def __init__(self, block, layers, num_classes=1000, drop_prob=0., block_size=5):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.dropblock = LinearScheduler(
            DropBlock2D(drop_prob=drop_prob, block_size=block_size),
            start_value=0.,
            stop_value=drop_prob,
            nr_steps=5e3
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):
        self.dropblock.step()  # increment number of iterations

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.dropblock(self.layer1(x))
        x = self.dropblock(self.layer2(x))
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x


def resnet9(**kwargs):
    return ResNetCustom(BasicBlock, [1, 1, 1, 1], **kwargs)

def resnet18(**kwargs):
    return ResNetCustom(BasicBlock, [2,2,2,2], **kwargs)

def logger(engine, model, evaluator, loader, pbar):
    evaluator.run(loader)
    metrics = evaluator.state.metrics
    avg_accuracy = metrics['accuracy']
    pbar.log_message(
        "Test Results[{:d}] - Avg accuracy: {:.5f}, drop_prob: {:.5f}".format(engine.state.epoch, avg_accuracy,
                                                                        model.dropblock.dropblock.drop_prob)
    )
    results.append(avg_accuracy)


if __name__ == '__main__':
    parser = configargparse.ArgumentParser()

    parser.add_argument('-c', '--config', required=False,
                        is_config_file=True, help='config file')
    parser.add_argument('--root', required=False, type=str, default='./data',
                        help='data root path')
    parser.add_argument('--dataset', required=False, type=str, default='cifar10',
                        help='dataset')
    parser.add_argument('--workers', required=False, type=int, default=4,
                        help='number of workers for data loader')
    parser.add_argument('--bsize', required=False, type=int, default=128,
                        help='batch size')
    parser.add_argument('--epochs', required=False, type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--lr', required=False, type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--drop_prob', required=False, type=float, default=0.,
                        help='dropblock dropout probability')
    parser.add_argument('--block_size', required=False, type=int, default=5,
                        help='dropblock block size')
    parser.add_argument('--device', required=False, default=None, type=int,
                        help='CUDA device id for GPU training')
    options = parser.parse_args()

    root = options.root
    bsize = options.bsize
    workers = options.workers
    epochs = options.epochs
    lr = options.lr
    drop_prob = options.drop_prob
    block_size = options.block_size
    device = 'cpu' if options.device is None \
        else torch.device('cuda:{}'.format(options.device))
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [123.675, 116.28, 103.53]],
                                     std=[x / 255.0 for x in [58.395, 57.12, 57.375]])
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    test_transform = transforms.Compose([
                                transforms.ToTensor(),
                                normalize])
    if options.dataset == 'cifar10':
        train_set = torchvision.datasets.CIFAR10(root=root, train=True,
                                                 download=True, transform=train_transform)
        test_set = torchvision.datasets.CIFAR10(root=root, train=False,
                                            download=True, transform=test_transform)
        num_classes = 10
    else:
        train_set = torchvision.datasets.CIFAR100(root=root, train=True,
                                                 download=True, transform=train_transform)
        test_set = torchvision.datasets.CIFAR100(root=root, train=False,
                                            download=True, transform=test_transform)
        num_classes = 100

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=bsize,
                                               shuffle=True, num_workers=workers)

    

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=bsize,
                                              shuffle=False, num_workers=workers)

    # classes = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # define model
    model = ResNet18(num_classes=num_classes, drop_prob=drop_prob, block_size=block_size)

    # define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                momentum=0.9, nesterov=True, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

    
    # create ignite engines
    trainer = create_supervised_trainer(model=model,
                                        optimizer=optimizer,
                                        loss_fn=criterion,
                                        device=device)

    evaluator = create_supervised_evaluator(model,
                                            metrics={'accuracy': Accuracy()},
                                            device=device)
    
    @trainer.on(Events.EPOCH_STARTED)
    def update_lr_schedulers(engine):
        scheduler.step()
    # ignite handlers
    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')

    pbar = ProgressBar()
    pbar.attach(trainer, ['loss'])

    trainer.add_event_handler(Events.EPOCH_COMPLETED, logger, model, evaluator, test_loader, pbar)

    # start training
    t0 = time.time()
    trainer.run(train_loader, max_epochs=epochs)
    t1 = time.time()
    print('Best Accuracy:', max(results))
    print('Total time:', t1 - t0)
