

local nn = require 'nn'
require 'cunn'
require 'cudnn'
require ('Model/LinearBatchCorr')

local M = {}

function M.netInit(opts, checkpoint, storePath)
	local net1, net2
	 
	-- Loading checkpoints or initialize a new network to train 
	if checkpoint then
		local net1Path = paths.concat(storePath, checkpoint.model1Name)
		local net2Path = paths.concat(storePath, checkpoint.model2Name)
		assert(paths.filep(net1Path), 'Saved net1 not found: ' .. net1Path)
		assert(paths.filep(net2Path), 'Saved net2 not found: ' .. net2Path)
		net1 = torch.load(net1Path):cuda()
		net1 = M.MultiGPU(net1, opts)
		net2 = torch.load(net2Path):cuda()

	else

		local RGB = torch.load('Model/resnet-50.t7')
		RGB:remove(11)
		local SN = torch.load('Model/resnet-50.t7')
		SN:remove(11)

		local s1 = nn.ParallelTable():add(RGB):add(SN)
		net1 = nn.Sequential():add(s1):add(nn.JoinTable(2, 2))
		print(net1)
		net1:cuda()
		net1 = M.MultiGPU(net1, opts)
		
		---------------------------------------------------------------------------------

		net2 = nn.Sequential()

		local s2
		if opts.CorrReg then			
			s2 = nn.LinearBatchCorr(4096, 51, opts.lambda,1, true, 'Gaussian')
		else
			s2 = nn.Linear(4096,51)
			s2.weight:normal(0,0.01)
			s2.bias:zero()
		end

		if opts.drop then
			net2:add(nn.Dropout(0.5))
		end
		net2:add(s2)

		print(net2)
		net2:cuda()
	end
	
	-- set up the training criterion module
	local criterion = nn.CrossEntropyCriterion() -- combining LogSoftMax and ClassNLLCriterion
	criterion:cuda() -- push into GPU
	
	return net1, net2, criterion    
end	

function M.MultiGPU(net, opts)
	-- remove the DataParallelTable for model replica, if any
	if torch.type(net) == 'nn.DataParallelTable' then
		net = net:get(1) -- the resulting 'net' is simply a contained network model in DataParallelTable
	end
	 
	-- optnet is an general library for reducing memory usage in neural networks
	if opts.optnet then
		local optnet = require 'optnet'
		local tmpsize = 224
		local tmpInput = {torch.zeros(4,3,tmpsize,tmpsize):cuda(),torch.zeros(4,3,tmpsize,tmpsize):cuda()}-- WHY USE '4' IMAGE SAMPLES HERE?
		optnet.optimizeMemory(net, tmpInput, {inplace = false, mode = 'training'})
	end
	 
	-- This is useful for reducing memory usage, but requires that all
	-- containers override backwards to call backwards recursively on submodules
	if opts.shareGradInput then
		M.shareGradInput(net)
	end
	 
	-- Set the CUDNN flags
	if opts.cudnnSetting == 'fastest' then
		cudnn.fastest = true
		cudnn.benchmark = true
	elseif opts.cudnnSetting == 'deterministic' then
		-- Use a deterministic convolution implementation
		net:apply(function(m) 
				if m.setMode then m:setMode(1, 1, 1) end
			end) -- 'apply' calls provided function on itself and all its child modules.
	end
	 
	-- Wrap the network with DataParallelTable, if using more than one GPU
	if opts.nGPU > 1 then
		local gpuIDs = torch.range(opts.gpuStartID, opts.gpuStartID+opts.nGPU-1):totable()
		local fastest, benchmark = cudnn.fastest, cudnn.benchmark
		 
		local netGPUReplicas = nn.DataParallelTable(1, true, true) -- splitting data at the first dimension 
									   -- flattenParams is true, getParameters() will be called on the replicated module
									   -- use NCCL to do inter-GPU communications
		netGPUReplicas:add(net, gpuIDs)	
		netGPUReplicas:threads(function() 
						local cudnn = require 'cudnn'
						require('Model/LinearBatchCorr')
						cudnn.fastest, cudnn.benchmark = fastest, benchmark
					end) -- use a seperate thread for each replica to do the function call in parallel
		netGPUReplicas.gradInput = nil -- to nil the gradInput for all the contained modules, BUT SEEMS NOT WORKING						
		 
		net = netGPUReplicas:cuda() -- push into GPUs		 
	end

	return net
end


function M.shareGradInput(model)
	local function sharingKey(m)
		local key = torch.type(m)
		if m.__shareGradInputKey then
			key = key .. ':' .. m.__shareGradInputKey
		end
		return key
	 end

	 -- Share gradInput for memory efficient backprop
	local cache = {}
	model:apply(function(m)
			local moduleType = torch.type(m)
			if torch.isTensor(m.gradInput) and moduleType ~= 'nn.ConcatTable' then
				local key = sharingKey(m)
				if cache[key] == nil then
					cache[key] = torch.CudaStorage(1)
				end
				m.gradInput = torch.CudaTensor(cache[key], 1, 0)
			end
		end)
	for i, m in ipairs(model:findModules('nn.ConcatTable')) do
		if cache[i % 2] == nil then
			cache[i % 2] = torch.CudaStorage(1)
		end
		m.gradInput = torch.CudaTensor(cache[i % 2], 1, 0)
	end
end

return M
