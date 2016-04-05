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

local HeuristicCMFA = {}

--
function HeuristicCMFA:new( ... )
   local o = {}
   setmetatable(o, self)
   self.__index = self
   o:__init(...)
   return o
end

--
function HeuristicCMFA:__init(cmfa)
   local n, s, p, k, f, N = cmfa:_setandgetDims()
   self.cmfa = cmfa

   -- Gx layer
   local GxM = nn.Sequential()
   GxM:add(nn.Linear(f, 20))
   GxM:add(nn.ReLU())
   GxM:add(nn.Linear(20, 50))
   GxM:add(nn.ReLU())
   GxM:add(nn.Linear(50, s * f))
   GxM:add(nn.ReLU())
   GxM:add(nn.CX(self.cmfa.factorLoading.Gm, s, p, f))

   self.GxM = nn.StochasticGradient(GxM, nn.SmoothL1Criterion())
   self.GxM.learningRate = 0.001
   self.GxM.maxIteration = 100

   -- Lz Layer
   local LzM = nn.Sequential()
   LzM:add(nn.Linear(f, 5))
   LzM:add(nn.ReLU())
   LzM:add(nn.Linear(5, s * k))
   LzM:add(nn.ReLU())
   LzM:add(nn.CX(self.cmfa.factorLoading.Lm, s, p, k))

   self.LzM = nn.StochasticGradient(LzM, nn.SmoothL1Criterion())
   self.LzM.learningRate = 0.001
   self.LzM.maxIteration = 100
end

--
function HeuristicCMFA:train(Xm, Zm, X_star) -- s x f x n, s x k x n, f x n
   local cmfa = self.cmfa
   local _, s, p, k, f = cmfa:_setandgetDims()
   local Lm, Lcov = cmfa.factorLoading:getL()
   local Gm, Gcov = cmfa.factorLoading:getG()
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
   trainset.label = torch.bmm(Gm, Xm):view(s*p, N):t() -- N x s*p
   trainset.data = nX_star

   self.GxM:train(trainset)
   print("Gx's training finished")

   print("Preparing dataset for learning Lz")
   trainset.label = torch.bmm(Lm, Zm):view(s*p, N):t() -- N x s*p
   trainset.data = nX_star

   self.LzM:train(trainset)
   print("Lz's training finished")
end

--
function HeuristicCMFA:forward(X_star) -- f x n
   local N = X_star:size(2)
   print(X_star:size())
   local _, s, p, k, f = self.cmfa:_setandgetDims()

   print("Normalizing X_star")
   local nX_star = X_star:t():clone()
   nX_star:add(self.hmean)
   nX_star:div(self.hstdv)

   local Gx = torch.zeros(s, p, N)
   local Lz = torch.zeros(s, p, N)
   for i = 1, N do
      local x = nX_star[{ i, {} }]
      Gx[{ {}, {}, i}] = self.GxM.module:forward(x):view(s, p)
      Lz[{ {}, {}, i}] = self.LzM.module:forward(x):view(s, p)
   end

   return Gx, Lz -- s x p x n
end

return HeuristicCMFA