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
   self.old = {}

   for epoch = 1, epochs do
      self.old.Xcov = self.conditional.Xcov:clone()
      self.old.Xm = self.conditional.Xm:clone()
      self.old.Zcov = self.hidden.Zcov:clone()
      self.old.Zm = self.hidden.Zm:clone()
      self.old.Lcov = self.factorLoading.Lcov:clone()
      self.old.Lm = self.factorLoading.Lm:clone()
      self.old.Gcov = self.factorLoading.Gcov:clone()
      self.old.Gm = self.factorLoading.Gm:clone()
      self.old.PsiI = self.hyper.PsiI:clone()
      self.old.Qs = self.hidden.Qs:clone()

      for subEpoch = 1, 5 do
         self:infer("Qz", Pt, Y)
         self:infer("HyperX", pause)
         self:infer("Qx", Pt, Y, X_star)
      end

      for subEpoch = 1, 5 do
         self:infer("QL", Pt, Y)
         self:infer("QG", Pt, Y)
      end

      self:infer("Qnu")
      self:infer("Qomega")

      self:infer("Qpi")
      self:infer("Qs", Pt, Y)

      self:infer("PsiI", Pt, Y)

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

      -- self:handleBirth(Y, Pt, X_star)
      collectgarbage()
      print(self.hidden.Qs:sum(1))
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
   for i = 1, self.s do
      local mean = self.factorLoading.Lm[i] * self.hidden.Zm[i] + self.factorLoading.Gm[i] * self.conditional.Xm[i]
      gnuplot.figure()
      gnuplot.scatter3(mean[1], mean[2], X_star[1])
      gnuplot.xlabel('Mux')
      gnuplot.ylabel('Muy')
      gnuplot.zlabel('X_star')
   end
end

function CFMA:infer(tr, ...)
   -- local t = os.clock()
   -- print(string.format("\n===== Infer%s =====\n", tr))
   self["infer"..tr](self, ...)
   -- print(string.format("\tt for %s = %f", tr, os.clock() - t))
   -- print("--------------------------------------------------")
   -- if self.pause == 1 then io.read() end
end


function CFMA:print(tr, ns)
   print("--------------------------------------------------")
   print(string.format("%s old = \n", tr))
   print(self.old[tr])
   print(string.format("%s after inference = \n", tr))
   print(self[ns][tr])
   print("--------------------------------------------------")
   if self.pause == 1 then io.read() end
end


return CFMA