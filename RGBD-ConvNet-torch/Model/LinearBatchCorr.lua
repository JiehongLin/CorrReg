local  LinearBatchCorr, Parent = torch.class('nn.LinearBatchCorr', 'nn.Module')
 
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

function LinearBatchCorr:__init(nFeaIn, nFeaOut, lambda, nReg, bias, initMethod)
	Parent.__init(self) 
	local bias = ((bias == nil) and true) or bias

	self.nFeaIn = nFeaIn
	self.nFeaOut = nFeaOut
	
	self.lambda = (-1)*lambda
	self.epsilon = 1e-4
	self.nReg = nReg or 1

	local nHalfFeaIn = nFeaIn/2
	if  nReg == 1 then
               	self.mask = torch.cat(torch.ones(1, nHalfFeaIn), 2*torch.ones(1, nHalfFeaIn), 2)
              else 
               	self.mask = getMaskSets(nReg, nFeaIn)
              end

	-- whole layer
	self.FC = nn.Linear(nFeaIn, nFeaOut, bias) 	

	if initMethod == 'xavierimproved' then
		local std = math.sqrt(2/nFeaOut) 
		self.FC.weight:normal(0,std)
	elseif initMethod == 'Gaussian' then
		self.FC.weight:normal(0, 0.01)
	else
		assert(false, 'LinearBatchCorr initMethod Error!')
	end
		
	if bias then 
		self.FC.bias:zero()
	end

end

function LinearBatchCorr:parameters()
	if self.FC.bias then
		return {self.FC.weight, self.FC.bias}, {self.FC.gradWeight, self.FC.gradBias}
	else
		return {self.FC.weight}, {self.FC.gradWeight}
	end
end

function LinearBatchCorr:gradient_batchcorr(Y1,Y2,nBatchSample)

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

function LinearBatchCorr:updateOutput(input)
	local output = self.FC:updateOutput(input)
	self.output:resize(output:size()):copy(output)
	return self.output
end


function LinearBatchCorr:updateGradInput(input, gradOutput)

	local nBatchSample = input:size(1) 
	local nFeaIn = input:size(2)
	local nFeaOut = self.FC.weight:size(1)

	local dCorrdX = torch.FloatTensor(input:size()):fill(0):cuda()
	local dCorrdW = torch.FloatTensor(self.FC.weight:size()):fill(0):cuda()

	-- part 1
	local FC1 = nn.Linear(nFeaIn*0.5, nFeaOut, false):cuda()
	-- part 2
	local FC2 = nn.Linear(nFeaIn*0.5, nFeaOut, false):cuda()

	for idx = 1,self.nReg do
	              	
            		local index1 = torch.range(1,nFeaIn)[torch.eq(self.mask[idx], 1):byte()]:long()
            		local index2 = torch.range(1,nFeaIn)[torch.eq(self.mask[idx], 2):byte()]:long()

		local X1 = input:index(2,index1)
		local X2 = input:index(2,index2)

		FC1.weight = self.FC.weight:index(2,index1)
		FC2.weight = self.FC.weight:index(2,index2)

		local Y1 =FC1:updateOutput(X1)
		local Y2 =FC2:updateOutput(X2)
		 
		local dCorrdY1,dCorrdY2=self:gradient_batchcorr(Y1,Y2,nBatchSample)

		FC1:zeroGradParameters()
		FC2:zeroGradParameters()

		FC1:backward(X1, dCorrdY1)
		FC2:backward(X2, dCorrdY2)

		dCorrdX:indexAdd(2,index1,FC1.gradInput)
		dCorrdX:indexAdd(2,index2,FC2.gradInput)
		dCorrdW:indexAdd(2,index1,FC1.gradWeight)
		dCorrdW:indexAdd(2,index2,FC2.gradWeight)

	end

	dCorrdX = dCorrdX/self.nReg
	dCorrdW = dCorrdW/self.nReg

	self.FC:backward(input, gradOutput) 
	self.FC.gradInput:add(self.lambda, dCorrdX)
	self.FC.gradWeight:add(self.lambda, dCorrdW)

	self.gradInput:resize(self.FC.gradInput:size()):copy(self.FC.gradInput)
	collectgarbage()
	return self.gradInput
end

