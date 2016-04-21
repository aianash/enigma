require 'nn'
require 'nngraph'

-----------------------------------------------
--[[ Constant Matrix Multiplication Layer ]]--
-----------------------------------------------

local CX, parent = torch.class('nn.CX', 'nn.Module')

--
function CX:__init(C, s, p, f) -- s x p x f
   parent.__init(self)

   self.C = C
   self.s = s
   self.p = p
   self.f = f
   self.gradInput = torch.Tensor()
end

--
function CX:updateOutput(input) -- s x f
   local s, p, f = self.s, self.p, self.f

   self.output:resize(s, p, 1)
   self.output:bmm(self.C, input:view(s, f, 1))
   self.output:resize(s * p)

   return self.output
end

--
function CX:updateGradInput(input, gradOutput)
   local s, p, f = self.s, self.p, self.f

   self.gradInput:resize(s, f, 1)
   self.gradInput:bmm(self.C:transpose(2, 3), gradOutput:view(s, p, 1))
   self.gradInput:resize(s * f)

   return self.gradInput
end

-----------------------
--[[ HeuristicVBCMFA ]]--
-----------------------
local HeuristicVBCMFA = klazz('enigma.cmfa.HeuristicVBCMFA')

--
function HeuristicVBCMFA:__init(vbcmfa)
   local n, S, p, k, f, N = vbcmfa:_setandgetDims()
   self.vbcmfa = vbcmfa

   -- Gx layer
   local GxM = nn.Sequential()
   GxM:add(nn.Linear(f, 20))
   GxM:add(nn.ReLU())
   GxM:add(nn.Linear(20, 50))
   GxM:add(nn.ReLU())
   GxM:add(nn.Linear(50, S * f))
   GxM:add(nn.ReLU())
   GxM:add(nn.CX(self.vbcmfa.factorLoading.Gm, S, p, f))

   self.GxM = nn.StochasticGradient(GxM, nn.SmoothL1Criterion())
   self.GxM.learningRate = 0.001
   self.GxM.maxIteration = 100

   -- Lz Layer
   local LzM = nn.Sequential()
   LzM:add(nn.Linear(f, 5))
   LzM:add(nn.ReLU())
   LzM:add(nn.Linear(5, S * k))
   LzM:add(nn.ReLU())
   LzM:add(nn.CX(self.vbcmfa.factorLoading.Lm, S, p, k))

   self.LzM = nn.StochasticGradient(LzM, nn.SmoothL1Criterion())
   self.LzM.learningRate = 0.001
   self.LzM.maxIteration = 100
end

--
function HeuristicVBCMFA:train(Xm, Zm, X_star) -- s x f x n, s x k x n, f x n
   local vbcmfa = self.vbcmfa
   local _, S, p, k, f = vbcmfa:_setandgetDims()
   local Lm, Lcov = vbcmfa.factorLoading:getL()
   local Gm, Gcov = vbcmfa.factorLoading:getG()
   local N = X_star:size(2)

   print("Normalizing X_star")
   local nX_star = X_star:t():clone()
   self.hmean = nX_star:mean()
   nX_star:add(self.hmean)
   self.hstdv = nX_star:std()
   nX_star:div(self.hstdv)

   local trainset = {}
   -- To prepare the dataset to be used with nn.StochasticGradient
   setmetatable(trainset,
       {__index = function(t, i)
                     return {t.data[i], t.label[i]}
                  end}
   )
   function trainset:size()
       return self.data:size(1)
   end

   print("Preparing dataset for learning Gx")
   trainset.label = torch.bmm(Gm, Xm):view(S*p, N):t() -- N x s*p
   trainset.data = nX_star

   self.GxM:train(trainset)
   print("Gx's training finished")

   print("Preparing dataset for learning Lz")
   trainset.label = torch.bmm(Lm, Zm):view(S*p, N):t() -- N x s*p
   trainset.data = nX_star

   self.LzM:train(trainset)
   print("Lz's training finished")
end

--
function HeuristicVBCMFA:forward(X_star) -- f x n
   local N = X_star:size(2)
   print(X_star:size())
   local _, S, p, k, f = self.vbcmfa:_setandgetDims()

   print("Normalizing X_star")
   local nX_star = X_star:t():clone()
   nX_star:add(self.hmean)
   nX_star:div(self.hstdv)

   local Gx = torch.zeros(S, p, N)
   local Lz = torch.zeros(S, p, N)
   for i = 1, N do
      local x = nX_star[{ i, {} }]
      Gx[{ {}, {}, i}] = self.GxM.module:forward(x):view(S, p)
      Lz[{ {}, {}, i}] = self.LzM.module:forward(x):view(S, p)
   end

   return Gx, Lz -- s x p x n
end

return HeuristicVBCMFA