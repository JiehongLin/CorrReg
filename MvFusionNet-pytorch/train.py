from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import os,shutil,json
import argparse
import time

from provider.ImgDataset import MultiviewImgDataset
from models.CorrRegResNet import resnet101

parser = argparse.ArgumentParser()
parser.add_argument("-train_path", type=str, default="modelnet40_images_new_12x/*/train")
parser.add_argument("-val_path", type=str, default="modelnet40_images_new_12x/*/test")
parser.add_argument("-bs", "--batchSize", type=int, help="Batch size for the second stage", default=16)# it will be *12 images in each batch for mvcnn
parser.add_argument("-num_views", type=int, help="number of views", default=12)
parser.add_argument("-lr", type=float, help="learning rate", default=1e-4)
parser.add_argument("-weight_decay", type=float, help="weight decay", default=1e-4)
parser.add_argument('--epochs', default=100, type=int, metavar='N', 
					help='number of total epochs to run (default: 100)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
					help='momentum (default: 0.9)')
parser.add_argument('--lr-decay-freq', default=30, type=float,
					metavar='W', help='learning rate decay (default: 30)')
parser.add_argument('--lr-decay', default=0.1, type=float,
					metavar='W', help='learning rate decay (default: 0.1)')
parser.add_argument('--lamda', default=0.0005, type=float,
					metavar='W', help='lamda')
parser.add_argument('--print-freq', '-p', default=10, type=int,
					metavar='N', help='print frequency (default: 10)')
parser.add_argument('-r', '--resume', default='', type=str, metavar='PATH',
					help='path to latest checkpoint (default: none)')
parser.add_argument("-no_pretraining", dest='no_pretraining', action='store_true')

parser.set_defaults(train=False)
args = parser.parse_args()


def create_folder(log_dir):
	# make summary folder
	if not os.path.exists(log_dir):
		os.mkdir(log_dir)


def train(train_loader, model, criterion, optimizer, epoch, saveddir):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()


	inputs_train = Variable(torch.FloatTensor(1).cuda())
	targets_train = Variable(torch.LongTensor(1).cuda())

	model.train()
	end = time.time()

	for i, (inputs, targets) in enumerate(train_loader):

		data_time.update(time.time() - end)
		inputs_train.data.resize_(inputs.size()).copy_(inputs)
		targets_train.data.resize_(targets.size()).copy_(targets)

		output = model(inputs_train)
		# print(output.size())
		loss = criterion(output, targets_train)
		# print(loss)

		prec = accuracy(output.data, targets_train.data)
		losses.update(loss.data[0], inputs.size(0))
		top1.update(prec, inputs.size(0))

		# compute gradient and do SGD step
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		# print(i)
		# print( i % args.print_freq)
		if i % args.print_freq == 0:
			print('Epoch: [{0}][{1}/{2}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				  'Prec {top1.val:.3f} ({top1.avg:.3f})\t'.format(
				   epoch, i, len(train_loader), batch_time=batch_time,
				   data_time=data_time, loss=losses, top1=top1))
	
	with open(os.path.join(saveddir, 'result.txt'), 'at') as f:
		f.write('epoch[{:3d}] train: {top1.avg:.3f}'.format(epoch, top1=top1))

def validate(val_loader, model, criterion, saveddir):	
	batch_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	model.eval()

	inputs_val = Variable(torch.FloatTensor(1).cuda(), volatile=True)
	targets_val = Variable(torch.LongTensor(1).cuda(), volatile=True)

	end = time.time()
	for i, (inputs, targets) in enumerate(val_loader):
		inputs_val.data.resize_(inputs.size()).copy_(inputs)
		targets_val.data.resize_(targets.size()).copy_(targets)
		output = model(inputs_val)
		# print(output)
		loss = criterion(output, targets_val)

		prec = accuracy(output.data, targets_val.data)
		losses.update(loss.data[0], inputs.size(0))
		top1.update(prec, inputs.size(0))			

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % args.print_freq == 0:
			print('Test: [{0}/{1}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
				   i, len(val_loader), batch_time=batch_time, loss=losses,top1=top1))
	
	with open(os.path.join(saveddir, 'result.txt'), 'at') as f:
		f.write('\t\ttest: {top1.avg:.3f}'.format(top1=top1))

	return top1.avg

def adjust_learning_rate(optimizer, epoch):
	lr = args.lr * (args.lr_decay ** (epoch // args.lr_decay_freq))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

def save_checkpoint(state, is_best, dir, filename='checkpoint.pth.tar'):
	torch.save(state, os.path.join(dir, filename))
	if is_best:
		shutil.copyfile(os.path.join(dir, filename), os.path.join(dir, 'model_best.pth.tar'))

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

def accuracy(output, target):
	"""Computes the precision@k for the specified values of k"""
	batch_size = target.size(0)
	_, pred = torch.max(output, 1)

	pred = pred.view(-1)
	target = target.view(-1)
	correct = torch.sum(pred.eq(target))

	return 100*correct/float(batch_size)
		
	
if __name__ == '__main__':

	pretraining = not args.no_pretraining
	log_dir = os.path.join('Exps/', 'Net_lamda'+str(args.lamda))
	create_folder('Exps')
	create_folder(log_dir)
	config_f = open(os.path.join(log_dir, 'config.json'), 'w')
	json.dump(vars(args), config_f)
	config_f.close()

	train_dataset = MultiviewImgDataset(args.train_path, scale_aug=False, rot_aug=False, num_views=args.num_views)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=False, num_workers=0)# shuffle needs to be false! it's done within the trainer
	val_dataset = MultiviewImgDataset(args.val_path, scale_aug=False, rot_aug=False, num_views=args.num_views)
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchSize, shuffle=False, num_workers=0)	
	print('num_train_files: '+str(len(train_dataset.filepaths)))
	print('num_val_files: '+str(len(val_dataset.filepaths)))

	model = resnet101(pretrained=pretraining, num_classes=40, lamda=args.lamda)
	model = torch.nn.DataParallel(model).cuda()
	criterion = nn.CrossEntropyLoss().cuda()
	optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))

	if args.resume:
		if os.path.isfile(args.resume):
			print("=> loading checkpoint '{}'".format(args.resume))
			checkpoint = torch.load(args.resume)
			start_epoch = checkpoint['epoch']
			best_prec = checkpoint['best_prec']
			model.load_state_dict(checkpoint['state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			print("=> loaded checkpoint '{}' (epoch {})"
				  .format(args.resume, checkpoint['epoch']))
		else:
			print("=> no checkpoint found at '{}'".format(args.resume))
	else:
		start_epoch = 1	
		best_prec = 0

	# cudnn.benchmark = True

	for epoch in range(start_epoch, args.epochs+1):
		adjust_learning_rate(optimizer, epoch)
		train(train_loader, model, criterion, optimizer, epoch, log_dir)
		prec = validate(val_loader, model, criterion, log_dir)

		is_best = prec > best_prec
		best_prec = max(prec, best_prec)
		save_checkpoint({
			'epoch': epoch + 1,
			'state_dict': model.state_dict(),
			'best_prec': best_prec,
			'optimizer' : optimizer.state_dict(),
		}, is_best, log_dir)
		
		print(' * Prec {:.3f},    Best {:.3f}\n'.format(prec, best_prec))
		if is_best:
			with open(os.path.join(log_dir, 'result.txt'), 'at') as f:
				f.write('\t\tbest: {:.3f}\n'.format(best_prec))
		else:
			with open(os.path.join(log_dir, 'result.txt'), 'at')as f:
				f.write('\n')

	print('Fisished!')


