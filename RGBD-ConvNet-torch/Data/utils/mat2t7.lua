
local matio = require 'matio'

local Dir = torch.load('./../dir/dir_all.t7')
local objectDir = Dir.ObjectDir
local n = #objectDir

print('File transform: \n')
for idx = 1,n do
	print(idx .. ' / ' .. n  .. ' : ' .. string.sub(objectDir[idx], 1, -5))
	local img = matio.load('./../data/' .. objectDir[idx], {'RGB', 'SN', 'target'})	
	local path = './../data/' .. string.sub(objectDir[idx], 1, -4) .. 't7'
	torch.save(path, img)
	os.execute('rm ./../data/' .. objectDir[idx])
end


