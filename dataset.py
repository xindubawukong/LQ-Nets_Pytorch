import torch
import torchvision
import os


def get_cifar10():
    path = './data/cifar10'
    train_dataset = torchvision.datasets.CIFAR10(
        root=path, train=True, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize((32, 32)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=path, train=False, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize((32, 32)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
    )
    return train_dataset, test_dataset


def get_imagenet():
    datadir = '/home1/data/ILSVRC2012' 
    traindir = os.path.join(datadir, 'train')
    valdir = os.path.join(datadir, 'val')
    train_dataset = torchvision.datasets.ImageFolder(
        traindir,
        torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
    )
    test_dataset = torchvision.datasets.ImageFolder(
        valdir,
        torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
    )
    return train_dataset, test_dataset


def get_dataset(name):
    if name == 'cifar10':
        return get_cifar10()
    elif name == 'imagenet':
        return get_imagenet()
    else:
        assert False, f'No dataset {name}.'


if __name__ == '__main__':
    train_dataset, test_dataset = get_dataset('cifar10')
    print(len(train_dataset), len(test_dataset))
