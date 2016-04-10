----------------------------------------------------------
-- conditional mixture of factor analyser
----------------------------------------------------------
require 'gnuplot'

local distributions = require 'distributions'
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
function CFMA:train(Mu, Pt, X_star, epochs)
   local n, s, p, k, d, N = self:_setandgetDims()
   self.old = {}

   self.conditional.Xm:copy(X_star:repeatTensor(s, 1, 1))

   for epoch = 1, epochs do
      print(string.format("\n-------------------------------------------------------\n"))
      print(string.format("epoch %d running...\n", epoch))

      if self.debug == 1 then
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
      end

      for convEpoch = 1, 20 do
         for subEpoch = 1, 10 do
            self:infer("Qz", Mu, Pt)
            self:infer("Qx", Mu, Pt, X_star)
         end

         for subEpoch = 1, 10 do
            self:infer("QL", Mu, Pt)
            self:infer("QG", Mu, Pt)
            self:infer("Qnu")
            self:infer("Qomega")
         end

         self:infer("Qs", Mu, Pt)
         self:infer("Qpi")

         if epoch % 2 == 0 then
            for subEpoch = 1, 5 do
               self:infer("PsiI", Mu, Pt)
               self:infer("HyperX", X_star)
            end
         end

         if self.debug == 1 then
            self:print("sigma_starI", "hyper")
            self:print("Zm", "hidden")
            self:print("Zcov", "hidden")
            self:print("Xm", "conditional")
            self:print("Xcov", "conditional")
            self:print("Lm", "factorLoading")
            self:print("Lcov", "factorLoading")
            self:print("Gm", "factorLoading")
            self:print("Gcov", "factorLoading")
            self:print("phim", "hidden")
            self:print("a", "factorLoading")
            self:print("b", "factorLoading")
            self:print("alpha", "factorLoading")
            self:print("beta", "factorLoading")
         end

      end

      self:infer("Qs", Mu, Pt)

      local F, dF = self:calcF(self.debug, Mu, Pt, X_star)
      print(string.format("F = %f\tdF = %f\n", F, dF))
      self.Fhist = self.Fhist:cat(torch.Tensor({F}))

      if epoch == 25 then
         self:handleBirth(Mu, Pt, X_star)
      end

      -- self:printFmatrix()
      collectgarbage()
   end

   self:plotFhist()
   -- self:plotPrediction(X_star)

   return self.F, self.factorLoading.Gm, self.factorLoading.Gcov, self.factorLoading.Lm, self.factorLoading.Lcov, self.hidden.Zm, self.conditional.Xm, self.hidden.Qs
end



function CFMA:printFmatrix()
   local Fmat = self.Fmatrix
   local percentageF = torch.zeros(Fmat:size())
   local s = Fmat:size(2)

   for i = 1, s do
      percentageF[{{}, i}] = Fmat[{{}, i}] * 100 / torch.sum(Fmat[{{}, i}])
   end

   print(string.format("Fmatrix = \n%s", self.Fmatrix))
   print(string.format("Fmatrix:sum(1) = \n%s", self.Fmatrix:sum(1)))
   print(string.format("F = %s", self.F))
   print(string.format("percentageF = \n%s", percentageF))
   print(string.format("phim = \n%s", self.hidden.phim))
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
   -- print("--------------------------------------------------")
   -- print(string.format("%s old = \n", tr))
   -- print(self.old[tr])
   print(string.format("%s after inference = \n", tr))
   print(self[ns][tr])
   -- print("--------------------------------------------------")
   if self.pause == 1 then io.read() end
end


return CFMA