
require 'image'

local M = {}

function M.Compose(transforms)
   return function(input)
      for _, transform in ipairs(transforms) do
         input = transform(input)
      end
      return input
   end
end

function M.ColorNormalize(meanstd)
   return function(img)
      img = img:clone()
      for i=1,3 do
         img[i]:add(-meanstd.mean[i])
         img[i]:div(meanstd.std[i])
      end
      return img
   end
end

-- Scales the smaller edge to size
function M.Scale(size, interpolation)
   interpolation = interpolation or 'bicubic'
   return function(input)
      local w, h = input:size(3), input:size(2)
      if (w <= h and w == size) or (h <= w and h == size) then
         return input
      end
      if w < h then
         return image.scale(input, size, h/w * size, interpolation)
      else
         return image.scale(input, w/h * size, size, interpolation)
      end
   end
end

-- Crop to centered rectangle
function M.CenterCrop(size)
   return function(input)
      local w1 = math.ceil((input:size(3) - size)/2)
      local h1 = math.ceil((input:size(2) - size)/2)
      return image.crop(input, w1, h1, w1 + size, h1 + size) -- center patch
   end
end

-- Random crop form larger image with optional zero padding
function M.RandomCrop(size, padding)
   padding = padding or 0

   return function(input)
      if padding > 0 then
         local temp = input.new(3, input:size(2) + 2*padding, input:size(3) + 2*padding)
         temp:zero()
            :narrow(2, padding+1, input:size(2))
            :narrow(3, padding+1, input:size(3))
            :copy(input)
         input = temp
      end

      local w, h = input:size(3), input:size(2)
      if w == size and h == size then
         return input
      end

      local x1, y1 = torch.random(0, w - size), torch.random(0, h - size)
      local out = image.crop(input, x1, y1, x1 + size, y1 + size)
      assert(out:size(2) == size and out:size(3) == size, 'wrong crop size')
      return out
   end
end


function M.HorizontalFlip(prob)
   return function(input)
      if torch.uniform() < prob then
         input = image.hflip(input)
      end
      return input
   end
end


return M
