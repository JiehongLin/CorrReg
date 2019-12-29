

local M = { }

function M.parse(arg)
	local cmd = torch.CmdLine() -- initialize the torch CmdLine class
	cmd:text() -- simply for writing an empty line
	cmd:text('Parameter settings used for this experiment')
	
	 ------------ General options --------------------
	cmd:option('-manualSeed',           0,                           'Manually set RNG seed')
	cmd:option('-nGPU',                       8,                           'Number of GPUs to use by default')
	cmd:option('-gpuStartID',              1,                           'This program will use the GPUs with IDs of [opts.gpuStartID, opts.gpuStartID+opts.nGPU-1]')
	cmd:option('-backend',                 'cudnn',                  'Options: cudnn | cunn')
	cmd:option('-cudnnSetting',         'deterministic',      'Options: fastest | default | deterministic')
	cmd:option('-expFolder',              'Exps',                     'Directory in which to save resulting experimental files/checkpoints')
	cmd:option('-shareGradInput',    'false',                    'Share gradInput tensors to reduce memory usage')
	cmd:option('-optnet',                    'false',                    'Use optnet to reduce memory usage')
	cmd:option('-nThreads',                8                              )
	
	------------- Training options --------------------

	cmd:option('-nEpoch',                  30,                          'Number of total epochs to run')
	cmd:option('-startEpoch',            1,                            'useful to specify a starting epoch to train')
	cmd:option('-batchSize',              128,                        'mini-batch size (1 = pure stochastic)')
	cmd:option('-lrTrain',                   0.01,                      'initial learning rate')
	cmd:option('-lrFinetune',            0.0001,                  'initial finetune learning rate')
	cmd:option('-alpha',                    0.1,                        'learning rate decay rate')
	cmd:option('-momentum',         0.9,                        'momentum')
	cmd:option('-weightDecay',       1e-4,                     'weight decay')
	cmd:option('-trials',                     1,                           'trials') 
	cmd:option('-lambda',                0.1                         ) 
	cmd:option('-drop',                     'false',                   'dropout') 
	cmd:option('-CorrReg',               'true',                    'CorrReg') 


	 cmd:text()
	
	local opts = cmd:parse(arg or {}) -- take tuning arguments from command line
	
	-- converting the command line arguments type of string to boolean
	opts.shareGradInput = opts.shareGradInput ~= 'false' 
	opts.optnet = opts.optnet ~= 'false'
	opts.drop = opts.drop ~= 'false'
	opts.CorrReg = opts.CorrReg ~= 'false'
	 
	if opts.shareGradInput and opts.optnet then
		cmd:error('error: cannot use both -shareGradInput and -optnet')
	end
	
	return opts
end

return M
