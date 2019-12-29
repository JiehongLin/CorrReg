
local optim = require 'optim'


local M = {}
local cnnTrain = torch.class('cnnTrain', M)

function cnnTrain:__init(net1, net2, criterion, optimState1, optimState2, opts)
	self.net1 = net1
	self.net2 = net2
	self.criterion = criterion
	self.optimState1 = optimState or {
		learningRate = opts.lrFinetune,
		momentum = opts.momentum,
		weightDecay = opts.weightDecay,
		nesterov = true, -- enables Nesterov momentum, Nesterov momentum requires a momentum and zero dampening
		learningRateDecay = 0.0,
		dampening = 0.0
	} 
	self.optimState2= optimState or {
		learningRate = opts.lrTrain,
		momentum = opts.momentum,
		weightDecay = opts.weightDecay,
		nesterov = true, -- enables Nesterov momentum, Nesterov momentum requires a momentum and zero dampening
		learningRateDecay = 0.0,
		dampening = 0.0
	}    
	self.opts = opts
	self.params1, self.gradParams1 = net1:getParameters() -- should only be called once for a network since storage may change                                   
	self.params2, self.gradParams2 = net2:getParameters()
end


function cnnTrain:epochTrain(epoch, dataLoader) -- training the network for one epoch
		
	self:learningRateSchedule(epoch)
	
	local timer = torch.Timer()
	local dataTimer = torch.Timer()
	
	local function feval1() -- function handler to be called by optimizer, upon value instantiation of self.gradParams and self.criterion
		return self.criterion.output, self.gradParams1
	end

	local function feval2() -- function handler to be called by optimizer, upon value instantiation of self.gradParams and self.criterion
		return self.criterion.output, self.gradParams2
	end

	local nIter = dataLoader:epochIterNum() -- no. of interations in this epoch 
	local Error, loss = 0.0, 0.0
	local nSample = 0 
	
	print('=> Training epoch # ' .. epoch)
   
	self.net1:training() -- setting mode of modules/sub-modules to training mode (train = true), useful for Dropout or BatchNormalization
	self.net2:training() 

	for iKey, batchSample in dataLoader:run() do 

		local dataTime = dataTimer:time().real
		self:copyBatchSample2GPU(batchSample)

		-- forward pass
		self.net1:forward(self.input)
		local output = self.net2:forward(self.net1.output):float()
		local batchSize = output:size(1)
		local currLoss = self.criterion:forward(self.net2.output, self.target)
	  
		self.net1:zeroGradParameters() -- clear gradients accumulation that is from accGradParameters
		self.net2:zeroGradParameters()

		-- backward pass
		self.criterion:backward(self.net2.output, self.target)
		self.net2:backward(self.net1.output, self.criterion.gradInput)
		self.net1:backward(self.input, self.net2.gradInput)
		
		-- updating parameters via SGD
		optim.sgd(feval1, self.params1, self.optimState1)
		optim.sgd(feval2, self.params2, self.optimState2)
		
		-- reporting errors (based on network parameters of the previous iteration) and other learning statistics
		local currErr = self:computeErrors(output, self.target, 1) -- returning averaged results over the mini-batch 
		
		Error = Error + currErr*batchSize
		loss = loss + currLoss*batchSize
		nSample = nSample + batchSize
		
		print((' | Epoch: [%d][%d/%d]    Time %.3f  Data %.3f  Loss %1.4f  Error %7.3f  '):format(
			epoch, iKey, nIter, timer:time().real, dataTime, currLoss, currErr))
		
		-- check that the storage didn't get changed due to an unfortunate getParameters call
		assert(self.params1:storage() == self.net1:parameters()[1]:storage())
		assert(self.params2:storage() == self.net2:parameters()[1]:storage())			

		timer:reset()
		dataTimer:reset()		
	  end
	
	  return Error / nSample,  loss / nSample
end
	
	
function cnnTrain:test(epoch, dataLoader) 

	local timer = torch.Timer()
	local dataTimer = torch.Timer()
	
	local nIter = dataLoader:epochIterNum() -- no. of interations
	local Error = 0.0
	local nSample = 0
	
	print('=> Performing testing on validation set after epoch # ' .. epoch)
	
	self.net1:evaluate() -- setting mode of modules/sub-modules to evaluation mode
	self.net2:evaluate()

	for iKey, batchSample in dataLoader:run() do 
		local dataTime = dataTimer:time().real		
		self:copyBatchSample2GPU(batchSample) -- Copy input and target to the GPU
		
		-- forward pass
		self.net1:forward(self.input)
		local output = self.net2:forward(self.net1.output):float()
		local batchSize = output:size(1)
		
		local currErr = self:computeErrors(output, self.target, 1) 
		Error = Error + currErr*batchSize
		nSample = nSample + batchSize
		
		print((' | Test: [%d][%d/%d]    Time %.3f  Data %.3f  Error %7.3f (%7.3f)  '):format(
			  epoch, iKey, nIter, timer:time().real, dataTime, currErr, Error/nSample))

		timer:reset()
		dataTimer:reset()
	end
	
	self.net1:training() -- resetting the mode of modules/sub-modules to training mode
	self.net2:training()

	print((' * Results after epoch # %d     Error: %7.3f  \n'):format(epoch, Error/nSample))	
	return Error/nSample
end	
		
	
function cnnTrain:computeErrors(output, target, nCrops)
	-- computation of errors are detached from the network
	-- running on CPU	
	
	if nCrops > 1 then
		-- sum over crops
		output = output:view(output:size(1) / nCrops, nCrops, output:size(2)):sum(2):squeeze(2)
	end

	-- computing the error rate
	local batchSize = output:size(1)

	local _ , predictions = output:float():sort(2, true) -- descending

	-- finding which predictions match the target
	local correct = predictions:eq(target:long():view(batchSize, 1):expandAs(output))

	-- error
	local Error = 1.0 - (correct:narrow(2, 1, 1):sum() / batchSize)

	return Error * 100
end
	
	
function cnnTrain:copyBatchSample2GPU(batchSample)
	local RGB = self.opts.nGPU == 1 and torch.CudaTensor() or cutorch.createCudaHostTensor() 
	local SN = self.opts.nGPU == 1 and torch.CudaTensor() or cutorch.createCudaHostTensor() 
	self.input = {}
	self.target = self.target or torch.CudaTensor()

	RGB:resize(batchSample.RGB:size()):copy(batchSample.RGB)
	SN:resize(batchSample.SN:size()):copy(batchSample.SN)
	
	self.input = {RGB, SN}
	self.target:resize(batchSample.target:size()):copy(batchSample.target)
end


function cnnTrain:learningRateSchedule(epoch)
	local learningRate1 = self.opts.lrFinetune
	local learningRate2 = self.opts.lrTrain
	local decay1 = 0
	--local decay1 =  epoch >= 11 and 2 or epoch >= 6 and 1 or 0
	self.optimState1.learningRate = learningRate1 * math.pow(self.opts.alpha, decay1)
	local decay2 =  epoch >= 21 and 2 or epoch >= 11 and 1 or 0
	self.optimState2.learningRate = learningRate2 * math.pow(self.opts.alpha, decay2)		

end

return M.cnnTrain
