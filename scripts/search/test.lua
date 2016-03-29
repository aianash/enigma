require 'torch'

local CMFA = require 'CMFA'

n = 10
N = 10
p = 2
s = 2
k = 3
d = 3

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

local sn = 2

local trials = 1

-- 50 iteration of random trials at algorithms
for it = 1, trials do
   local cmfa = CMFA:new(cfg)

   Y = torch.randn(sn, p, n)
   P = torch.randn(n, sn)
   P = P + torch.abs(torch.min(P))
   Pt = torch.cdiv(P, P:sum(2):repeatTensor(1, sn))

   cmfa:train(Y, Pt, 100)
end