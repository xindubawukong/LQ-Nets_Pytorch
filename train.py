import torch
import argparse
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

from dataset import get_dataset
from utils import *
from vgg import *


def inference(epoch, net, dataloader, optimizer, device, is_train=False):
    if is_train:
        net.train()
    else:
        net.eval()
    disp_interval = 50
    loss_func = torch.nn.CrossEntropyLoss()

    loss_avg = AverageMeter()
    top1_avg = AverageMeter()
    top5_avg = AverageMeter()

    start_time = time.time()

    for step, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = net(images)
        top1, top5 = get_accuracy(outputs, labels)
        loss = loss_func(outputs, labels)

        loss_avg.update(loss.item(), images.shape[0])
        top1_avg.update(top1.item(), images.shape[0])
        top5_avg.update(top5.item(), images.shape[0])

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if step > 0 and step % disp_interval  == 0:
            duration = float(time.time() - start_time)
            example_per_second = images.size(0) * disp_interval / duration
            lr = optimizer.param_groups[0]['lr']
            print("epoch:[%.3d]  step: %d  top1: %f  top5: %f  loss: %.6f  fps: %.3f  lr: %.5f " %
                  (epoch, step, top1_avg.avg, top5_avg.avg, loss.item(), example_per_second, lr)
            )
            start_time = time.time()
    
    return top1_avg.avg, top5_avg.avg, loss_avg.avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, help='cpu or cuda', default='cpu')
    parser.add_argument('--max_epoch', type=int, help='max epochs', default=10)
    parser.add_argument('--seed', type=int, help='random seed', default=0)
    parser.add_argument('--batch_size', type=int, help='batch size', default=64)
    parser.add_argument('--dataset', type=str, help='cifar10 or imagenet', default='cifar10')
    args = parser.parse_args()
    print('args:', args)

    assert args.device in ['cpu', 'cuda']
    torch.manual_seed(args.seed)

    if not os.path.exists('log'):
        os.mkdir('log')
    log_path = os.path.join('log', f'{time.strftime("%Y%m%d%H%M%S", time.localtime())}')
    os.mkdir(log_path)

    train_dataset, test_dataset = get_dataset('cifar10')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    net = vgg11_bn(pretrained=True, num_classes=1000).to(args.device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)

    history = []
    for epoch in range(args.max_epoch):
        lr_scheduler.step()
        train_result = inference(epoch, net, train_dataloader, optimizer, args.device, is_train=True)
        test_result = inference(epoch, net, test_dataloader, optimizer, args.device, is_train=False)
        print('train_result: top1: {}  top5: {}  loss: {}'.format(*train_result))
        print('test_result: top1: {}  top5: {}  loss: {}'.format(*test_result))
        history.append({
            'epoch': epoch,
            'train_result': train_result,
            'test_result': test_result,
        })
        info = {
            'history': history,
            'state_dict': net.state_dict(),
        }
        torch.save(info, os.path.join(log_path, f'epoch_{epoch}.pth'))
    
    print('Bye~')


if __name__ == '__main__':
    main()