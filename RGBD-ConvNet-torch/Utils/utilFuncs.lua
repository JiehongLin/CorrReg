
local M = {}

function M.writeErrsToFile(fpath, epoch, Error, trnLoss, mode)
	-- mode ~ 'train' | 'val' 
	local file

	if paths.filep(fpath) then
		file = io.open(fpath, 'a') -- append
	else
		file = io.open(fpath, 'w') -- create a new one
	end

	if mode == 'train' then
		file:write(string.format('Training-Epoch:%d  Error:%7.3f  Loss %1.4f', epoch, Error, trnLoss))
	elseif mode == 'val' then
		file:write(string.format('       Testing-Epoch:%d  Error:%7.3f\n', epoch, Error))
	else
		error('The mode of writing erros to file must be either train / val !')
	end

	file:close()
end

return M