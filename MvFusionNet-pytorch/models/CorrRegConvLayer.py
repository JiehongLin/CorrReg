import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.autograd import Function
import ipdb
from torch.nn.modules.utils import _pair

# Inherit from Function
class CorrConvFunction(Function):

	# Note that both forward and backward are @staticmethods
	@staticmethod
	# bias is an optional argument
	def forward(ctx, input, weight, bias=None, stride=1, padding=0, lamda=0.005):
		# ipdb.set_trace()
		ctx.save_for_backward(input, weight, bias)
		ctx.lamda = lamda
		ctx.stride = stride
		ctx.padding = padding
		# output = nn.functional.conv2d(input, weight, bias=bias, stride=stride, padding=padding)
		output = nn.functional.conv2d(Variable(input), Variable(weight), bias=Variable(bias), stride=stride, padding=padding)
		return output.data

	# This function has only a single output, so it gets only one gradient
	@staticmethod
	def backward(ctx, grad_output):

		input, weight, bias = ctx.saved_variables
		lamda = ctx.lamda
		stride = ctx.stride
		padding = ctx.padding
		HalfFeaIn = int(input.size(1) *0.5)
		BatchSize = input.size(0) 
		eps = 1e-4
		grad_input = grad_weight = grad_bias = None

		input1 = input[:, :HalfFeaIn]
		input2 = input[:, HalfFeaIn:]

		X1 = Variable(input1.data.clone(), requires_grad=True)
		X2 = Variable(input2.data.clone(), requires_grad=True)

		W1 = weight[:, :HalfFeaIn]
		W2 = weight[:, HalfFeaIn:]

		net1 = nn.Conv2d(W1.size(1), W1.size(0), [W1.size(2), W1.size(3)], bias=None, stride=stride, padding=padding).cuda()
		net2 = nn.Conv2d(W2.size(1), W2.size(0), [W2.size(2), W2.size(3)], bias=None, stride=stride, padding=padding).cuda()

		net1.weight.data.copy_(W1.data)
		net2.weight.data.copy_(W2.data)
		Y1 = net1(X1)
		Y2 = net2(X2)

		Offset1 = Y1 - torch.mean(Y1, 0, keepdim=True).expand_as(Y1)
		Offset2 = Y2 - torch.mean(Y2, 0, keepdim=True).expand_as(Y2)
		CrossVar = torch.sum(Offset1 * Offset2, 0, keepdim=True)
		AbsVar1 = torch.sum(Offset1, 0, keepdim=True)
		AbsVar2 = torch.sum(Offset2, 0, keepdim=True)

		Sigma1 = torch.sum(Offset1**2, 0, keepdim=True)
		Sigma2 = torch.sum(Offset2**2, 0, keepdim=True)

		tmpExp_I = torch.pow(Sigma1*Sigma2+eps, -0.5)
		tmpExp_II = (-0.5) * torch.pow(tmpExp_I, 3)

		dCorrdSigma1 = tmpExp_II * Sigma2 * CrossVar
		dCorrdSigma2 = tmpExp_II * Sigma1 * CrossVar

		dCorrdMu1 = (-1) * AbsVar2 * tmpExp_I + (-2) * dCorrdSigma1 * AbsVar1
		dCorrdMu2 = (-1) * AbsVar1 * tmpExp_I + (-2) * dCorrdSigma2 * AbsVar2

		dCorrdY1 = Offset2 * tmpExp_I.expand_as(Y1) + dCorrdMu1.expand_as(Y1)/BatchSize + 2*Offset1*dCorrdSigma1.expand_as(Y1)
		dCorrdY2 = Offset1 * tmpExp_I.expand_as(Y2) + dCorrdMu2.expand_as(Y2)/BatchSize + 2*Offset2*dCorrdSigma2.expand_as(Y2)

		Y1.backward(dCorrdY1)
		Y2.backward(dCorrdY2)

		dCorrdX = torch.cat((X1.grad, X2.grad), 1)
		dCorrdW = torch.cat((net1.weight.grad, net2.weight.grad), 1)

		net = nn.Conv2d(weight.size(1), weight.size(0), [weight.size(2), weight.size(3)], stride=stride, padding=padding).cuda()
		net.weight.data.copy_(weight.data)
		net.bias.data.copy_(bias.data)
		new_input = Variable(input.data.clone(), requires_grad=True)
		output = net(new_input)
		output.backward(grad_output)
		# ipdb.set_trace()	

		if ctx.needs_input_grad[0]:
			grad_input = new_input.grad - lamda * dCorrdX
		if ctx.needs_input_grad[1]:
			grad_weight = net.weight.grad - lamda * dCorrdW
		if bias is not None and ctx.needs_input_grad[2]:
			grad_bias = net.bias.grad
			
		return grad_input, grad_weight, grad_bias, None, None, None

class CorrConv(nn.Module):
	def __init__(self, input_features, output_features, kernel_size, stride=1, padding=0, bias=True, lamda=0.005):
		super(CorrConv, self).__init__()
		self.input_features = input_features
		self.output_features = output_features
		self.kernel_size = kernel_size
		self.stride = stride
		self.padding = padding
		self.lamda = lamda

		self.weight = nn.Parameter(torch.Tensor(output_features, input_features, kernel_size, kernel_size))
		if bias:
			self.bias = nn.Parameter(torch.Tensor(output_features))
		else:
			# You should always register all possible parameters, but the
			# optional ones can be None if you want.
			self.register_parameter('bias', None)

		# Not a very smart way to initialize weights
		self.weight.data.normal_(0, 0.01)
		if bias is not None:
			self.bias.data.zero_()

	def forward(self, input):
		# See the autograd section for explanation of what happens here.
		# output = torch.nn.functional.conv2d(input, self.weight)
		# print(output.requires_grad)
		return CorrConvFunction.apply(input, self.weight, self.bias, self.stride, self.padding, self.lamda)

	def extra_repr(self):
		# (Optional)Set the extra information about this module. You can test
		# it by printing an object of this class.
		return 'in_features={}, out_features={}, bias={}'.format(
			self.in_features, self.out_features, self.bias is not None
		)