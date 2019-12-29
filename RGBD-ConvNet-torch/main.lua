
local initTimer = torch.Timer()

require 'torch'
require 'cutorch'
require 'paths'
require 'optim'
require 'nn'
print(('The time of requring torch packages is: %.3f'):format(initTimer:time().real))

----------------------------------------------------------------------------------

initTimer:reset()
local optsArgParse = require 'optsArgParse'
local opts = optsArgParse.parse(arg)
print(opts)

local dataLoader = require('Data/utils/dataLoader') 
local netBuilder = require('Model/NetInit')
local cnnTrainer = require 'cnnTrain'
local checkpoint = require('Utils/checkpoint')
local utils = require('Utils/utilFuncs')
print(('The time of requring project packages is: %.3f'):format(initTimer:time().real))

----------------------------------------------------------------------------------

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1) 
torch.manualSeed(opts.manualSeed)
cutorch.manualSeedAll(opts.manualSeed)

local prefix = 'RGBSN_trial_' .. opts.trials  ..  '_CorrReg_' .. tostring(opts.CorrReg) .. '_drop_' .. tostring(opts.drop) .. '_lambda_' .. opts.lambda ..
		'_nEpoch_' .. opts.nEpoch ..  '_batchSize_' .. opts.batchSize ..  '_lrTrain_' .. opts.lrTrain .. '_lrFinetune_' .. opts.lrFinetune .. '_weightDecay_' .. opts.weightDecay 
local storePath = paths.concat(opts.expFolder, prefix)
print('File Dir:  ' .. storePath)

---------------------------------------------------------------------------------

-- creating callback data dir loading functions
initTimer:reset()
local trnLoader, valLoader = dataLoader.create(opts)
print(('The time of creating dataLoader is: %.3f'):format(initTimer:time().real)) 

----------------------------------------------------------------------------------

-- loading the latest training checkpoint if it exists
initTimer:reset()
local latestpoint, optimState1,  optimState2= checkpoint.latest(storePath, opts) -- returning nil if not existing 
print(('The time of loading latest checkpoint is: %.3f'):format(initTimer:time().real)) 

----------------------------------------------------------------------------------

-- loading the latest or create a new network model
initTimer:reset()
local net1, net2, criterion = netBuilder.netInit(opts, latestpoint, storePath)
print(('The time of initializing network model is: %.3f'):format(initTimer:time().real))

----------------------------------------------------------------------------------

-- initialize the trainer, which handles training loop and evaluation on the validation set
local trainer = cnnTrainer(net1, net2, criterion, optimState1, optimState2, opts) 

-- start training 
local startEpoch = latestpoint and latestpoint.epoch + 1 or opts.startEpoch
local bestError = math.huge
local statsFPath = paths.concat(opts.expFolder, 'stats_' .. prefix .. '.txt')

for epoch = startEpoch, opts.nEpoch do

	-- Train for a single epoch
	local trnError, trnLoss = trainer:epochTrain(epoch, trnLoader)
	utils.writeErrsToFile(statsFPath, epoch, trnError, trnLoss, 'train')

	-- Run model on validation set
	local testError = trainer:test(epoch, valLoader)
	utils.writeErrsToFile(statsFPath, epoch, testError, nil, 'val')

	local bestModelFlag = false
	if testError < bestError then
		bestModelFlag = true -- true to save the up to now best model
		bestError = testError
	end

	checkpoint.save(net1, net2, trainer.optimState1, trainer.optimState2, epoch, bestModelFlag, storePath, opts)
end

