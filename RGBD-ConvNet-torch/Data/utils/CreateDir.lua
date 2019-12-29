
require 'paths'

local ObjectDir = paths.dir('./../data')
table.remove(ObjectDir,1)    --remove '.'
table.remove(ObjectDir,1)    --remove '..'

local ObjectName = {}
local ClassName = {}


for idx, object in ipairs(ObjectDir) do

	local firstIndex1,lastIndex1,firstIndex2,lastIndex2,cname,oname
	firstIndex1,lastIndex1,cname = string.find(object,"(.+)_%d+_%d+.mat")
	firstIndex2,lastIndex2,oname = string.find(object,"(.+)_%d+.mat")
	
	table.insert(ClassName,cname)
	table.insert(ObjectName,oname)	
end

local nImgAll = #ObjectDir
local set = torch.ones(10,nImgAll)       -- 1 for train , 2  for  test  

local file = io.open("trials.txt","r")
local t1,t2,t3,t4,t5,t6,t7,t8,t9,t10 = {},{},{},{},{},{},{},{},{},{}
local trials = {t1,t2,t3,t4,t5,t6,t7,t8,t9,t10}

local i = 1
for line in file:lines() do
	local m = i%54
	local n = math.ceil(i/54)
	if m>1 and m<53 then
		local firstIndex3,lastIndex3,name
		firstIndex3,lastIndex3,name = string.find(line,"(.+_%d+)%s*") 
		table.insert(trials[n],name)
	end
	i = i+1
end


for  try = 1,10  do
	for objectIndex, object in ipairs(ObjectName) do
		for ClassIndex = 1,51 do
			if trials[try][ClassIndex] == object then
				set[try][objectIndex] = 2
			end
		end
	end
end

os.execute('mkdir ./../dir')
local Dir = {ObjectDir=ObjectDir,set=set}
torch.save('./../dir/dir_all.t7',Dir)

for trials = 1,10 do
	
	local set = Dir.set[trials]
	local Index1 = torch.range(1,nImgAll)[torch.eq(set, 1)]:long()
	local Index2 = torch.range(1,nImgAll)[torch.eq(set, 2)]:long()

	local TrnDir={}
	for idx = 1, Index1:size(1) do
		TrnDir[idx] = string.sub(Dir.ObjectDir[Index1[idx]], 1, -4) .. 't7'
	end

	local ValDir={}
	for jdx = 1, Index2:size(1) do
		ValDir[jdx] = string.sub(Dir.ObjectDir[Index2[jdx]], 1, -4) .. 't7'
	end

	print('trials ' .. trials .. ' : ' .. Index1:size(1) .. ' images for training, ' .. Index2:size(1) .. ' images for testing\n')
	torch.save('./../dir/trial' .. trials .. '_trnDir.t7', TrnDir)
	torch.save('./../dir/trial' .. trials .. '_valDir.t7', ValDir)

end

