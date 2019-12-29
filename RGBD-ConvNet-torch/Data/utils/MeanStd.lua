
os.execute('mkdir ./../meanstd')

print('compute rgb meanstd: \n')
os.execute('mkdir ./../meanstd/RGB')
for trials = 1,10 do
	print('trials' .. trials)
	local dir = torch.load('./../dir/trial' .. trials .. '_trnDir.t7')
	local img = torch.zeros(#dir, 3, 256, 256):float()
	for Idx = 1,#dir do
		 local path = './../data/' .. dir[Idx]
		 info = torch.load(path)
		 img[Idx] =  info.RGB:float()
	end
	local mean = torch.Tensor(3)
	local std = torch.Tensor(3)

	mean[1] = torch.mean(img[{{}, 1, {}, {}}])
	mean[2] = torch.mean(img[{{}, 2, {}, {}}])
	mean[3] = torch.mean(img[{{}, 3, {}, {}}])
	std[1] = torch.std(img[{{}, 1, {}, {}}])
	std[2] = torch.std(img[{{}, 2, {}, {}}])
	std[3] = torch.std(img[{{}, 3, {}, {}}])

	colorNormalize = {mean=mean, std = std}
	torch.save('./../meanstd/RGB/trial' .. trials .. '.t7' , colorNormalize) 
end

print('compute sn meanstd: \n')
os.execute('mkdir ./../meanstd/SN')
for trials = 1,10 do
	print('trials' .. trials)
	local dir = torch.load('./../dir/trial' .. trials .. '_trnDir.t7')
	local img = torch.zeros(#dir, 3, 256, 256):float()
	for Idx = 1,#dir do
		 local path = './../data/' .. dir[Idx]
		 info = torch.load(path)
		 img[Idx] =  info.SN:float()
	end
	local mean = torch.Tensor(3)
	local std = torch.Tensor(3)

	mean[1] = torch.mean(img[{{}, 1, {}, {}}])
	mean[2] = torch.mean(img[{{}, 2, {}, {}}])
	mean[3] = torch.mean(img[{{}, 3, {}, {}}])
	std[1] = torch.std(img[{{}, 1, {}, {}}])
	std[2] = torch.std(img[{{}, 2, {}, {}}])
	std[3] = torch.std(img[{{}, 3, {}, {}}])

	colorNormalize = {mean=mean, std = std}
	torch.save('./../meanstd/SN/trial' .. trials .. '.t7' , colorNormalize) 
end





