
require  'paths'
require 'image'
local t = require('Data/utils/transforms')

local M = {}
local imdb = torch.class('imdb', M)

function imdb.create(opts, trnValSplit) 

	local SplitDir
	if trnValSplit=='train' then
		SplitDir = torch.load('Data/dir/trial' .. opts.trials .. '_trnDir.t7')
	else
		SplitDir = torch.load('Data/dir/trial' .. opts.trials .. '_valDir.t7')
	end

	local imdbTrnVal = M.imdb(SplitDir, opts, trnValSplit)
	return imdbTrnVal
end

function imdb:__init(SplitDir, opts, trnValSplit)
	self.SplitDir = SplitDir
	print('data size (' .. trnValSplit .. ') :  ' .. #SplitDir)
	self.trnValSplit = trnValSplit
	self.trials = opts.trials
end

function imdb:size()
	return #self.SplitDir
end

function imdb:get(i)
	local dir = self.SplitDir[i]
	local path ='Data/data/' .. dir
	assert(paths.filep(path), path .. ': image path does not exist!')
	return torch.load(path)
end


function imdb:preprocess_RGB()
	local meanstd = torch.load('Data/meanstd/RGB/trial' ..  self.trials .. '.t7')
	if self.trnValSplit == 'train' then
		return t.Compose{
				t.RandomCrop(224),
				t.ColorNormalize(meanstd),
				t.HorizontalFlip(0.5),
		}
	elseif self.trnValSplit == 'val' then
		local Crop =  t.CenterCrop
		return t.Compose{
				t.ColorNormalize(meanstd),
				Crop(224),
		}
	else
		error('invalid split: ' .. self.split)
	end
end

function imdb:preprocess_SN()
	--local meanstd = torch.load('Data/meanstd/SN/trial' ..  self.trials .. '.t7')
	if self.trnValSplit == 'train' then
		return t.Compose{
				t.RandomCrop(224),
				--t.ColorNormalize(meanstd),
				t.HorizontalFlip(0.5),
		}
	elseif self.trnValSplit == 'val' then
		local Crop =  t.CenterCrop
		return t.Compose{
				--t.ColorNormalize(meanstd),
				Crop(224),
		}
	else
		error('invalid split: ' .. self.split)
	end
end

return M.imdb
