import torch
import torch.nn as nn
import argparse
import time
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

from dataset import get_dataset
from utils import *
import vgg
import vgg_quant
import resnet
import resnet_quant


lr_count = 0


def adjust_learning_rate(optimizer, history):
    if len(history) > 3 and history[-1]['test_result'][0] < min([history[i - 4]['test_result'][0] for i in range(3)]):
        global lr_count
        if lr_count < 3:
            lr_count += 1
            lr = optimizer.param_groups[0]['lr'] * 0.1
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

    for name, p in net.named_parameters():
        print(name, p.size(), p.device)

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
            for p in list(net.parameters()):
                if hasattr(p,'org'):
                    p.data.copy_(p.org)
            optimizer.step()
            for p in list(net.parameters()):
                if hasattr(p,'org'):
                    p.org.copy_(p.data)
            for m in net.modules():
                if hasattr(m, 'record'):
                    if len(m.record) > 0:
                        m.basis.data = torch.cat(m.record).mean(dim=0).view(m.num_filters, m.nbit)
                        m.record = []
        
        if step > 0 and step % disp_interval  == 0:
            duration = float(time.time() - start_time)
            example_per_second = images.size(0) * disp_interval / duration
            lr = optimizer.param_groups[0]['lr']
            print("epoch:[%.3d]  step: %d  top1: %f  top5: %f  loss: %.6f  fps: %.3f  lr: %.6f " %
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
    parser.add_argument('--lr', type=float, help='init learning rate', default=0.01)
    args = parser.parse_args()
    print('args:', args)

    assert args.device in ['cpu', 'cuda']
    if args.device == 'cuda' and args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.manual_seed(args.seed)

    if not os.path.exists('log'):
        os.mkdir('log')
    log_path = os.path.join('log', f'{time.strftime("%Y%m%d%H%M%S", time.localtime())}')
    os.mkdir(log_path)

    train_dataset, test_dataset = get_dataset(args.dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    num_classes = 10 if args.dataset == 'cifar10' else 1000
    if args.model == 'vgg':
        net = vgg_quant.vgg11_bn(pretrained=False, num_classes=num_classes, w_bit=args.w_bit, a_bit=args.a_bit)
    else:
        net = resnet_quant.resnet18(pretrained=False, num_classes=num_classes, w_bit=args.w_bit, a_bit=args.a_bit)
    if args.device == 'cuda':
        net = nn.DataParallel(net)
    net = net.to('cuda:0')
    for name, p in net.named_parameters():
            print(name, p.size(), p.device)
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
        })
        info = {
            'history': history,
            'state_dict': net.state_dict(),
            'args': args,
        }
        torch.save(info, os.path.join(log_path, f'epoch_{epoch}.pth'))
    
    print(f'All results saved to {log_path}.\nBye~')


if __name__ == '__main__':
    main()