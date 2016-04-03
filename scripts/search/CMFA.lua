----------------------------------------------------------
-- conditional mixture of factor analyser
----------------------------------------------------------
require 'gnuplot'

local VBCMFA = require 'VBCMFA'

local CFMA = {}

local parent = VBCMFA:_factory()
setmetatable(CFMA, parent)
parent.__index = parent


function CFMA:new(...)
   local o = {}
   setmetatable(o, self)
   self.__index = self
   o:__init(...)
   return o
end


function CFMA:__init(cfg)
   self.debug = cfg.debug
   self.pause = cfg.pause
   self.Fhist = torch.Tensor(1):fill(-1 / 0)
   parent:__init(cfg)
end


-- Mu  means of factor analyzers of intent
-- Pti responsibilities
function CFMA:train(Mu, Pti, X_star, epochs)
   local n, s, p, k, d, N = self:_setandgetDims()

   for epoch = 1, epochs do
      for subEpoch = 1, 10 do
         self:infer("Qz", Pt, Y)
         self:infer("HyperX", pause)
         self:infer("Qx", Pt, Y, X_star)
      end

      for subEpoch = 1, 10 do
         self:infer("QL", Pt, Y)
         self:infer("QG", Pt, Y)
      end

      for subEpoch = 1, 10 do
         self:infer("Qnu")
         self:infer("Qomega")
      end

      for subEpoch = 1, 10 do
         self:infer("Qpi")
         self:infer("Qs", Pt, Y)
      end

      for subEpoch = 1, 10 do
         self:infer("PsiI", Pt, Y)
      end

      if self.debug == 1 then
         self:print("Zm", "hidden")
         self:print("Zcov", "hidden")
         self:print("Xm", "conditional")
         self:print("Xcov", "conditional")
         self:print("Lm", "factorLoading")
         self:print("Lcov", "factorLoading")
         self:print("Gm", "factorLoading")
         self:print("Gcov", "factorLoading")
      end

      local F, dF = self:calcF(self.debug, Y, Pt, X_star)
      self.Fhist = self.Fhist:cat(torch.Tensor({F}))

      self:handleBirth(Y, Pt, X_star)

      collectgarbage()
   end

   self:plotFhist()
   self:plotPrediction(X_star)
   print(string.format("Number of components = %d", self.s))
end


function CFMA:plotFhist()
   gnuplot.figure()
   gnuplot.grid(true)
   gnuplot.plot(self.Fhist)
   gnuplot.xlabel("Epochs")
   gnuplot.ylabel("Lower Bound F")
end


function CFMA:plotPrediction(X_star)
   local mean = self.factorLoading.Lm[1] * self.hidden.Zm[1] + self.factorLoading.Gm[1] * self.conditional.Xm[1]
   gnuplot.figure()
   gnuplot.grid(true)

   gnuplot.scatter3(mean[1], mean[2], X_star[1])
end

function CFMA:infer(tr, ...)
   local t = os.clock()
   -- print(string.format("\n===== Infer%s =====\n", tr))
   self["infer"..tr](self, ...)
   -- print(string.format("\tt for %s = %f", tr, os.clock() - t))
   -- print("--------------------------------------------------")
   if self.pause == 1 then io.read() end
end


function CFMA:print(tr, ns)
   print(string.format("%s after inference = \n", tr))
   print(self[ns][tr])
   if self.pause == 1 then io.read() end
end


return CFMA