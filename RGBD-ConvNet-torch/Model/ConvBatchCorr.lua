local  ConvBatchCorr, Parent = torch.class('nn.ConvBatchCorr', 'nn.Module')

 local function getMaskSets(nReg, nFeaIn)
	local nHalfFeaIn = nFeaIn/2
	local tmpmask =  torch.cat(torch.ones(1, nHalfFeaIn), 2*torch.ones(1, nHalfFeaIn), 2)
	local mask = torch.zeros(nReg, nFeaIn)

	for idx = 1,nReg do
		local tmpidx = torch.randperm(nFeaIn):long()
		mask[idx] = tmpmask:index(2,tmpidx)
	end
	return mask
 end
  
function ConvBatchCorr:__init(nFeaIn, nFeaOut, kerW, kerH, strideW, strideH, padW, padH, lambda, nReg, bias, initMethod)
	Parent.__init(self) 
	local bias = ((bias == nil) and true) or bias

	self.nFeaIn = nFeaIn
	self.nFeaOut = nFeaOut
	self.kerW = kerW
	self.kerH = kerH

	self.strideW = strideW or 1
	self.strideH = strideH or 1
	self.padW = padW or 0
	self.padH = padH or self.padW
	
	self.lambda = (-1)*lambda 
	self.epsilon = 1e-4
	self.nReg = nReg or 1 
	
	--mask
	local nHalfFeaIn = nFeaIn * 0.5
	if nReg == 1 then
		self.mask =  torch.cat(torch.ones(1, nHalfFeaIn), 2*torch.ones(1, nHalfFeaIn), 2)
	else
		self.mask = getMaskSets(nReg, nFeaIn)
	end
  
	-- whole layer
	self.conv = nn.SpatialConvolution(nFeaIn, nFeaOut, kerW, kerH, strideW, strideH, padW, padH)

	if initMethod == 'xavierimproved' then
		local std = math.sqrt(2/(kerW * kerH * nFeaOut)) 
		self.conv.weight:normal(0,std)
	elseif initMethod == 'Gaussian' then
		self.conv.weight:normal(0, 0.01)
	else
		assert(false, 'ConvBatchCorr initMethod Error!')
	end

	if bias then
		self.conv.bias:zero()
	else
		self.conv:noBias()
	end

end


function ConvBatchCorr:parameters()
	if self.conv.bias then
		return {self.conv.weight, self.conv.bias}, {self.conv.gradWeight, self.conv.gradBias}
	else
		return {self.conv.weight}, {self.conv.gradWeight}
	end
end


function ConvBatchCorr:gradient_batchcorr(Y1,Y2,nBatchSample)

	local Mu1 = torch.mean(Y1,1)  --internal variable Mu1
	local Mu2 = torch.mean(Y2,1)  --internal variable Mu2	   

	local Offset1 = Y1-torch.expand(Mu1,Y1:size())
	local Offset2 = Y2-torch.expand(Mu2,Y2:size())
	local CrossVar = torch.sum(torch.cmul(Offset1,Offset2),1)
	local AbsVar1 = torch.sum(Offset1,1)
	local AbsVar2 = torch.sum(Offset2,1)
			
	local sigma1 = torch.sum(torch.pow(Offset1,2),1) 
	local sigma2 = torch.sum(torch.pow(Offset2,2),1)
			
	local tmpExp_I = torch.pow(torch.cmul(sigma1,sigma2)+self.epsilon, -0.5)
	local tmpExp_II = torch.pow(tmpExp_I,3)*(-0.5)
				  
	local dCorrdSigma1 = torch.cmul(torch.cmul(tmpExp_II, sigma2), CrossVar)
	local dCorrdSigma2 = torch.cmul(torch.cmul(tmpExp_II, sigma1), CrossVar)
	
	local dCorrdMu1 =  torch.cmul(AbsVar2,tmpExp_I) *(-1) + torch.cmul(dCorrdSigma1,AbsVar1)*(-2)
	local dCorrdMu2 =  torch.cmul(AbsVar1,tmpExp_I) *(-1) + torch.cmul(dCorrdSigma2,AbsVar2)*(-2)

	-- dCorrdY1
	local dCorrdY1 = torch.cmul(Offset2, torch.expand(tmpExp_I,Y1:size()))+torch.expand(dCorrdMu1/nBatchSample,Y1:size())+torch.cmul(Offset1*2, torch.expand(dCorrdSigma1,Y1:size()))
		
	-- dCorrdY2	   
	local dCorrdY2 = torch.cmul(Offset1, torch.expand(tmpExp_I,Y2:size()))+torch.expand(dCorrdMu2/nBatchSample,Y2:size())+torch.cmul(Offset2*2, torch.expand(dCorrdSigma2,Y2:size()))
		
	return  dCorrdY1,dCorrdY2

end


function ConvBatchCorr:updateOutput(input)
	local output = self.conv:updateOutput(input)
	self.output:resize(output:size()):copy(output)
	return self.output
end


function ConvBatchCorr:updateGradInput(input, gradOutput)

	local nBatchSample = input:size(1) 

	local dCorrdX = torch.FloatTensor(input:size()):fill(0):cuda()
	local dCorrdW = torch.FloatTensor(self.conv.weight:size()):fill(0):cuda()

	----part1
	local conv1 = nn.SpatialConvolution(self.nFeaIn*0.5, self.nFeaOut, self.kerW, self.kerH, self.strideW, self.strideH, self.padW, self.padH):cuda()
	conv1:noBias()
	
	-----part2
	local conv2 = nn.SpatialConvolution(self.nFeaIn*0.5, self.nFeaOut, self.kerW, self.kerH, self.strideW, self.strideH, self.padW, self.padH):cuda()
	conv2:noBias()

	for idx = 1,self.nReg do		

            		local index1 = torch.range(1,self.nFeaIn)[torch.eq(self.mask[idx], 1):byte()]:long()
            		local index2 = torch.range(1,self.nFeaIn)[torch.eq(self.mask[idx], 2):byte()]:long()

		local X1 = input:index(2,index1)
		local X2 = input:index(2,index2)

		conv1.weight = self.conv.weight:index(2,index1)
		conv2.weight = self.conv.weight:index(2,index2)

		local Y1 = conv1:updateOutput(X1)
		local Y2 = conv2:updateOutput(X2)

		local dCorrdY1,dCorrdY2=self:gradient_batchcorr(Y1,Y2,nBatchSample)
		
		conv1:zeroGradParameters()
		conv2:zeroGradParameters()
			
		conv1:backward(X1, dCorrdY1)
		conv2:backward(X2, dCorrdY2)	

		dCorrdX:indexAdd(2,index1,conv1.gradInput)
		dCorrdX:indexAdd(2,index2,conv2.gradInput)
		dCorrdW:indexAdd(2,index1,conv1.gradWeight)
		dCorrdW:indexAdd(2,index2,conv2.gradWeight)
	end
	
	dCorrdX = dCorrdX/self.nReg
	dCorrdW = dCorrdW/self.nReg

	self.conv:backward(input, gradOutput) 
	self.conv.gradInput:add(self.lambda, dCorrdX)
	self.conv.gradWeight:add(self.lambda, dCorrdW)

	self.gradInput:resize(self.conv.gradInput:size()):copy(self.conv.gradInput)
	collectgarbage()
	return self.gradInput
end


