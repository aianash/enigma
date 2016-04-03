require 'torch'
require 'gnuplot'

local CMFA = require 'CMFA'

n = 1000
N = 1000
p = 2
s = 1
k = 1
d = 1

local cfg = {
   batchSize = n,
   numComponents = s,
   outputVectorSize = p,
   factorVectorSize = k,
   inputVectorSize = d,
   datasetSize = N,
   debug = 0,
   pause = 0
}

-------------------------------------------------------------
----------------- data generation ---------------------------
-------------------------------------------------------------
local sn = 1

local X_star = torch.randn(n, d)
local L = torch.randn(p, k)
local G = torch.randn(p, d)

Y = torch.zeros(sn, n, p)
Ysn = Y[1]

for i = 1, n do
   local z = torch.zeros(k)
   z[i % k + 1] = 1
   if i % 3 == 0 then
      Ysn[i] = G * X_star[i] + L * z + torch.randn(p)
   elseif i % 3 == 1 then
      Ysn[i] = G * (X_star[i] + 25) + L * z + torch.randn(p)
   else
      Ysn[i] = G * (X_star[i] + 50) + L * z + torch.randn(p)
   end
end

Y = Y:transpose(2, 3)
Pt = torch.ones(n, sn)

gnuplot.scatter3(Y[1][1], Y[1][2], X_star[{{}, 1}])
-------------------------------------------------------------
-------------------------------------------------------------

local trials = 1

-- 50 iteration of random trials at algorithms
for it = 1, trials do
   print(string.format("Trail number %d running...", it))
   local cmfa = CMFA:new(cfg)

   cmfa:train(Y, Pt, X_star:t():contiguous(), 50)
end