
local M = {}

function M.latest(storePath, opts)
    	local latestPath = paths.concat(storePath, 'latest.t7')
    	if not paths.filep(latestPath) then
    		if not paths.filep(storePath) then
            			os.execute('mkdir ' ..  storePath)
            	end
        	return nil
    	end
	
	-- loading from the latest checkpoint
	print('=> Loading checkpoint from latestPath ')
    	local latest = torch.load(latestPath)
	local optimState1 = torch.load(paths.concat(storePath, latest.optim1Name))
	local optimState2 = torch.load(paths.concat(storePath, latest.optim2Name))

	return latest, optimState1, optimState2
end


function M.save(net1, net2, optimState1, optimState2, epoch, bestModelFlag, storePath, opts)

	local function netRecursiveCopy(net)
  		local copyNet = {}
  		for iKey, moduleVal in pairs(net) do
		    	if type(moduleVal) == 'table' then
		    		copyNet[iKey] = netRecursiveCopy(moduleVal)
			else
		    		copyNet[iKey] = moduleVal 
			end
		end    

		if torch.typename(net) then
		    	torch.setmetatable(copyNet, torch.typename(net))
		end

		return copyNet
	end


    	-- don't save the DataParallelTable for easier loading on other machines
    	if torch.type(net1) == 'nn.DataParallelTable' then
        	net1 = net1:get(1)
    	end
    	if torch.type(net2) == 'nn.DataParallelTable' then
        	net2 = net2:get(1)
    	end
	
	-- create a clean copy on the CPU without modifying the original network
    	net1= netRecursiveCopy(net1):float():clearState() -- Clears intermediate module states such as output, gradInput and others
	net2= netRecursiveCopy(net2):float():clearState() -- Clears intermediate module states such as output, gradInput and others

	local model1Name = 'net1_latest.t7'
	local model2Name = 'net2_latest.t7'
	local optim1Name = 'optimState1_latest.t7'
	local optim2Name = 'optimState2_latest.t7'
	
	torch.save(paths.concat(storePath, model1Name), net1)
	torch.save(paths.concat(storePath, model2Name), net2)
	torch.save(paths.concat(storePath, optim1Name), optimState1)
	torch.save(paths.concat(storePath, optim2Name), optimState2)

	-- for latest checking
	torch.save(paths.concat(storePath, 'latest.t7'), {epoch = epoch, model1Name = model1Name, model2Name = model2Name, optim1Name = optim1Name, optim2Name = optim2Name})
	
    	if bestModelFlag then -- bestModelFlag is specified when calling this save function 
        		torch.save(paths.concat(storePath,  'net1' .. '_' .. opts.trials .. '_best.t7'), net1)
        		torch.save(paths.concat(storePath,  'net2' .. '_' .. opts.trials .. '_best.t7'), net2)
    	end
   
end

return M