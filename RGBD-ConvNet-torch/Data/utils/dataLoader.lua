
local imdbTrnVal =require('Data/utils/RGBSNinit')
local threads = require 'threads'
threads.serialization('threads.sharedserialize')

local M = {}
local dataLoader = torch.class('dataLoader', M)

function dataLoader.create(opts)
	local trnValDataLoaders = {}
	for iKey, value in ipairs({'train', 'val'}) do
		local imdb = imdbTrnVal.create(opts, value)
		trnValDataLoaders[iKey] = M.dataLoader(imdb, opts, value)
	end	
	return table.unpack(trnValDataLoaders)
end


function dataLoader:__init(imdb, opts, trnValSplit)
	local manualSeed = opts.manualSeed -- for use to initialize a different manual seed for each thread of data loading
	local function init()
		require('Data/utils/RGBSNinit')
	end
	local function main(threadid) -- threadid ~ [1, opts.nThreads]
		torch.manualSeed(manualSeed + threadid) -- initialzing a different random seed for each thread
		torch.setnumthreads(1) -- WHY DO THIS?
		_G.imdb = imdb -- the imdb whose mode of either 'train' or 'val' is already specified
		_G.preprocess_RGB = imdb:preprocess_RGB() -- doing pre-processing including data augmentation 
		_G.preprocess_SN = imdb:preprocess_SN()
		return imdb:size() 
	end
	-- initialize a pool of threads
	local threadPool, nImgInIMDB = threads.Threads(opts.nThreads, init, main) -- no. of images in different threads
	self.trials = opts.trials
	self.threadPool = threadPool
	self.nImgInIMDB = nImgInIMDB[1][1]
	self.batchSize = opts.batchSize 
	self.split = trnValSplit
end

function dataLoader:epochIterNum()
	if self.split == 'train' then 
		return math.floor(self.nImgInIMDB / self.batchSize) -- the no. of iterations per epoch 
	elseif self.split == 'val' then
		return math.ceil(self.nImgInIMDB / self.batchSize)
	else
		assert(false)
	end
end

function dataLoader:run() -- callback function for data loading during training/inference
	local threadPool = self.threadPool 
	local nImgInIMDB, batchSize = self.nImgInIMDB, self.batchSize
	local tmpindices = torch.randperm(nImgInIMDB) -- reshuffled indices of 'train' or 'val' samples
	
	local batchImgSamples = {} -- variable for hosting a mini-batch of image samples (data and other info.)
	local idx = 1 -- the index of ('train' or 'val') image samples, ranging in [1, nImgInIMDB] 
	local iter = 0 -- the mini-batch iteration over an epoch 
	
	local nIter = self:epochIterNum() 
	local countIter = 1

	local function enqueue()
		-- distributing the jobs of loading and pre-processing an epoch of mini-batches of image samples over a pool of threads 
		while countIter <= nIter and threadPool:acceptsjob()	do -- acceptsjob() return true if the pool of thread queues is not full
			local tmpbatchindices = tmpindices:narrow(1, idx, math.min(batchSize, nImgInIMDB-idx+1))
						-- distributing the following jobs of mini-batches to the pool of threads
			threadPool:addjob(
				function(tmpindices) -- callback function, executed on each threads
					local nImgSample = tmpindices:size(1)	
					local target = torch.IntTensor(nImgSample) -- variable for hosting training targets/labels of image samples
					local RGB, SN, tmpsizes

					for iKey, idxValue in ipairs(tmpindices:totable()) do
						local currImgSample = _G.imdb:get(idxValue) 
						local currRGB = _G.preprocess_RGB(currImgSample.RGB) -- do data augmentation on the fly
						local currSN = _G.preprocess_SN(currImgSample.SN)
						if not RGB then
							tmpsizes = currRGB:size():totable() 
							RGB = torch.FloatTensor(nImgSample, table.unpack(tmpsizes))
						end
						if not SN then
							tmpsizes = currSN:size():totable() 
							SN = torch.FloatTensor(nImgSample, table.unpack(tmpsizes))
						end
						RGB[iKey]:copy(currRGB)
						SN[iKey]:copy(currSN)
						target[iKey] = currImgSample.target
					end
					collectgarbage() -- automtic management/freeing of garbage memory oppupied by the preceding operations 
					return {RGB=RGB, SN=SN, target = target}					
				end, 
				
				function(_batchImgSamples_) -- endcallback function whose argument is from the return of callback function, executed on the main thread, 
					batchImgSamples = _batchImgSamples_ -- pass the mini-batch of image samples to the main thread
				end, 
				
				tmpbatchindices    -- arguments of callback function  
			)
			
			idx = idx + batchSize
			countIter = countIter+1			
		end		
	end	

	local function loop()
		enqueue() -- loading and processing a mini-batch of image samples over a free thread 
		if not threadPool:hasjob() then -- true if there is still any job unfinished
			return nil -- finish the 'loop' function when all jobs are done  
		end 
		
		threadPool:dojob() -- to tell the main thread to execute the next endcallback in the queue
		if threadPool:haserror() then
			threadPool:synchronize()
		end
		
		enqueue() -- SHOULD WE DO ANOTHER ROUND HERE
		iter = iter + 1
		
		return iter, batchImgSamples 
	end
	
	return loop
end

return M.dataLoader
