import torch
import torch.nn as nn
import argparse
import time
import os

from dataset import get_dataset
from utils import *
import vgg
import vgg_quant
import resnet
import resnet_quant


def adjust_learning_rate(optimizer, history):
    if not hasattr(adjust_learning_rate, 'lr_count'):
        adjust_learning_rate.lr_count = 0
    if not hasattr(adjust_learning_rate, 'last_time'):
        adjust_learning_rate.last_time = 0
    if len(history) > 3 and history[-1]['test_result'][0] < min([history[i - 4]['test_result'][0] for i in range(3)]):
        if adjust_learning_rate.lr_count < 2 and adjust_learning_rate.last_time + 5 <= history[-1]['epoch']:
            print('Bring down learning rate.')
            adjust_learning_rate.lr_count += 1
            adjust_learning_rate.last_time = history[-1]['epoch']
            lr = optimizer.param_groups[0]['lr'] * 0.2
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr


def inference(epoch, net, dataloader, optimizer, device, is_train=False):
    if is_train:
        net.train()
    else:
        net.eval()
    disp_interval = 10
    loss_func = torch.nn.CrossEntropyLoss()

    loss_avg = AverageMeter()
    top1_avg = AverageMeter()
    top5_avg = AverageMeter()

    start_time = time.time()

    for step, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        net = net.to(device)
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
            for m in net.modules():
                if hasattr(m, 'record'):
                    if len(m.record) > 0:
                        new_basis = torch.cat(m.record).mean(dim=0).view(m.num_filters, m.nbit)
                        new_basis = new_basis.to(m.basis.device)
                        m.basis.data = m.basis.data * 0.9 + new_basis.data * 0.1
                        m.record = []
        
        if step > 0 and step % disp_interval  == 0:
            duration = float(time.time() - start_time)
            example_per_second = images.size(0) * disp_interval / duration
            lr = optimizer.param_groups[0]['lr']
            print("epoch[%.3d]  step: %d  top1: %f  top5: %f  loss: %.6f  fps: %.3f  lr: %.6f " %
                  (epoch, step, top1_avg.avg, top5_avg.avg, loss.item(), example_per_second, lr)
            )
            start_time = time.time()
    
    return top1_avg.avg, top5_avg.avg, loss_avg.avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, help='cpu or cuda', default='cpu')
    parser.add_argument('--gpu', type=str, help='comma separated list of GPU(s) to use.')
    parser.add_argument('--model', type=str, help='vgg or resnet', default='vgg')
    parser.add_argument('--dataset', type=str, help='cifar10 or imagenet', default='cifar10')
    parser.add_argument('--max_epoch', type=int, help='max epochs', default=10)
    parser.add_argument('--seed', type=int, help='random seed', default=0)
    parser.add_argument('--batch_size', type=int, help='batch size', default=64)
    parser.add_argument('--w_bit', type=int, help='weight quant bits', default=0)
    parser.add_argument('--a_bit', type=int, help='activation quant bits', default=0)
    parser.add_argument('--method', type=str, help='QEM or BP', default='QEM')
    parser.add_argument('--lr', type=float, help='init learning rate', default=0.01)
    args = parser.parse_args()
    print('args:', args)

    assert args.device in ['cpu', 'cuda']
    if args.device == 'cuda' and args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.manual_seed(args.seed)

    assert args.method in ['QEM', 'BP']
    assert args.w_bit <= 4
    assert args.a_bit <= 4

    if not os.path.exists('log'):
        os.mkdir('log')
    log_path = os.path.join('log', f'{time.strftime("%Y%m%d%H%M%S", time.localtime())}')
    os.mkdir(log_path)

    train_dataset, test_dataset = get_dataset(args.dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    num_classes = 10 if args.dataset == 'cifar10' else 1000
    if args.model == 'vgg':
        net = vgg_quant.vgg11_bn(pretrained=False, num_classes=num_classes, w_bit=args.w_bit, a_bit=args.a_bit, method=args.method)
    else:
        net = resnet_quant.resnet18(pretrained=False, num_classes=num_classes, w_bit=args.w_bit, a_bit=args.a_bit, method=args.method)
    if args.device == 'cuda':
        net = nn.DataParallel(net)
    net = net.to(args.device)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    history = []
    for epoch in range(args.max_epoch):
        adjust_learning_rate(optimizer, history)
        train_result = inference(epoch, net, train_dataloader, optimizer, args.device, is_train=True)
        with torch.no_grad():
            test_result = inference(epoch, net, test_dataloader, optimizer, args.device, is_train=False)
        print('train_result: top1: {}  top5: {}  loss: {}'.format(*train_result))
        print('test_result: top1: {}  top5: {}  loss: {}'.format(*test_result))
        history.append({
            'epoch': epoch,
            'train_result': train_result,
            'test_result': test_result,
            'lr': optimizer.param_groups[0]['lr'],
        })
        info = {
            'history': history,
            'state_dict': net.state_dict(),
            'args': args,
        }
        torch.save(info, os.path.join(log_path, f'epoch_{epoch}.pth'))
        with open(os.path.join(log_path, 'aaa.txt'), 'w') as f:
            f.write(f'args: {args}\n')
            for t in history:
                f.write(str(t) + '\n')
    
    print(f'All results saved to {log_path}.\nBye~')


if __name__ == '__main__':
    main()