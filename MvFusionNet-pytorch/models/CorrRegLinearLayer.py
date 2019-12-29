import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Function

# Inherit from Function
class CorrLinearFunction(Function):

	# Note that both forward and backward are @staticmethods
	@staticmethod
	# bias is an optional argument
	def forward(ctx, input, weight, bias=None, lamda=0.005):
		ctx.save_for_backward(input, weight, bias)
		ctx.lamda = lamda
		output = input.mm(weight.t())
		if bias is not None:
			output += bias.unsqueeze(0).expand_as(output)
		return output

	# This function has only a single output, so it gets only one gradient
	@staticmethod
	def backward(ctx, grad_output):

		input, weight, bias = ctx.saved_variables
		lamda = ctx.lamda
		HalfFeaIn = int(input.size(1) *0.5)
		BatchSize = input.size(0) 
		eps = 1e-4
		grad_input = grad_weight = grad_bias = None

		X1 = input[:, :HalfFeaIn]
		X2 = input[:, HalfFeaIn:]

		W1 = weight[:, :HalfFeaIn]
		W2 = weight[:, HalfFeaIn:]

		Y1 = X1.mm(W1.t())
		Y2 = X2.mm(W2.t())

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

		gradX1 = dCorrdY1.mm(W1)
		gradX2 = dCorrdY2.mm(W2)
		gradW1 = dCorrdY1.t().mm(X1)
		gradW2 = dCorrdY2.t().mm(X2)

		dCorrdX = torch.cat((gradX1, gradX2), 1)
		dCorrdW = torch.cat((gradW1, gradW2), 1)

		if ctx.needs_input_grad[0]:
			grad_input = grad_output.mm(weight) - lamda * dCorrdX
		if ctx.needs_input_grad[1]:
			grad_weight = grad_output.t().mm(input) - lamda * dCorrdW
		if bias is not None and ctx.needs_input_grad[2]:
			grad_bias = grad_output.sum(0).squeeze(0)

		return grad_input, grad_weight, grad_bias, None

class CorrLinear(nn.Module):
	def __init__(self, input_features, output_features, bias=True, lamda=0.005):
		super(CorrLinear, self).__init__()
		self.input_features = input_features
		self.output_features = output_features
		self.lamda = lamda

		self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
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
		return CorrLinearFunction.apply(input, self.weight, self.bias, self.lamda)

	def extra_repr(self):
		# (Optional)Set the extra information about this module. You can test
		# it by printing an object of this class.
		return 'in_features={}, out_features={}, bias={}'.format(
			self.in_features, self.out_features, self.bias is not None
		)