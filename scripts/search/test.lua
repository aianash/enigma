require 'torch'
require 'gnuplot'
require 'distributions'
require 'image'
require 'nn'

local CMFA = require 'CMFA'

torch.setdefaulttensortype('torch.FloatTensor')

----------------------------------------------
-- generate 3D spiral dataset
-- n : number of datapoints
-- k : number of means
-- note: n / k should be an integer
----------------------------------------------
function generateSpiralData(n, k)
   local z = torch.linspace(-2 * math.pi, 2 * math.pi, k)
   local x = z:clone():cos()
   local y = z:clone():sin()

   local nx = torch.zeros(1)
   local ny = torch.zeros(1)
   local nz = torch.zeros(1)

   for i = 1, k do
      local rnd = torch.Tensor(n / k, 3)
      distributions.mvn.rnd(rnd, torch.Tensor({x[i], y[i], z[i]}), torch.eye(3) * 0.1)
      nx = nx:cat(rnd[{{}, 1}])
      ny = ny:cat(rnd[{{}, 2}])
      nz = nz:cat(rnd[{{}, 3}])
   end

   nx = nx[{{2, n + 1}}]
   ny = ny[{{2, n + 1}}]
   nz = nz[{{2, n + 1}}]

   nx = nx:cat(nx + 5)
   ny = ny:cat(ny + 5)
   nz = nz:cat(nz + 5)

   local Y = nx:view(1, 2 * n):clone()
   Y = Y:cat(ny:view(1, 2 * n), 1)
   local X_star = nz
   return Y, X_star:view(2 * n, 1)
end


-- normalize dataset
function normalize(data)
   local normKernel = image.gaussian1D(7)
   local norm = nn.SpatialContrastiveNormalization(3, normKernel)
   local batchSize = 200
   local N = data:size(1)
   for i = 1, N, batchSize do
      local batch = math.min(N, i + batchSize) - i
      local nImgs = norm:forward(data:narrow(1, i, batch))
      data:narrow(1, i, batch):copy(nImgs)
   end
end


local trainDataset = torch.load("collar-glimpses-train.bin")
local I = trainDataset.data
local M, g, c, h, w = I:size(1), I:size(2), I:size(3), I:size(4), I:size(5)
local N = M * g
local h_star, w_star = 3, 3

normalize(I:view(M * g, c, h, w))

Y = torch.zeros(h * w, N)
X_star = torch.zeros(h_star * w_star, N)

for i = 1, M do
   for j = 1, g do
      local n = (i - 1) * g + j
      local src = I[{i, j, {}, {}, {}}]
      local grey = image.rgb2y(src)

      image.save("./glimpses/"..tostring(n).."-1.ppm", src)
      image.save("./glimpses/"..tostring(n).."-2.pgm", grey)

      Y[{ {}, n }]:copy(grey)
      X_star[{ {}, n }]:copy(image.scale(grey, w_star, h_star))
   end
end

print(string.format("Size of Y = %s", Y:size()))
print(string.format("Size of X_star = %s", X_star:size()))

local s = 2
local k = 3
local p = h * w
local f = h_star * w_star
local sn = 1

local Pt = torch.ones(N, sn)
Y = Y:view(sn, Y:size(1), Y:size(2))

local cmfa = CMFA:new{
   batchSize = N,
   numComponents = s,
   outputVectorSize = p,
   factorVectorSize = k,
   inputVectorSize = f, -- 16 x 16 inout image
   datasetSize = N,
   delay = 1,
   forgettingRate = 0.6,
   debug = 0,
   pause = 0,
   hardness = 1
}

cmfa:train(Y, Pt, X_star:contiguous(), 20)
