local VBCMFA = require 'scripts.feature.vbcmfa'
require 'nn'
require 'nngraph'


--------------------------------------------------
--[[ CMFA ]]--
-- Conditonal Mixture of Factor Analysers
--------------------------------------------------
local CMFA = {}

local parent = VBCMFA:_factory()
setmetatable(CMFA, parent)
parent.__index = parent

function CMFA:new( ... )
   local o = {}
   setmetatable(o, self)
   self.__index = self
   o:__init(...)
   return o
end

function CMFA:__init( ... )
   parent:__init( ... )
   local n, s, p, k, f, N = self:_setandgetDims()

   self.Xm_N = torch.zeros(s, f, N)
   self.Zm_N = torch.randn(s, k, N)
   self.Qs_N = torch.ones(N, s) / s
end

--
function CMFA:batchtrain(Ybatch, X_starbatch, batchperm, batchIdx, epochs)
   local n, s, p, k, f, N = self:_setandgetDims()
   assert(self:_checkDimensions(Ybatch, p, n))
   assert(self:_checkDimensions(X_starbatch, f, n))

   print(X_starbatch:index(2, torch.linspace(1, n, 20):floor():long()))
   print(X_starbatch:std(2))
   self:check(Ybatch:contiguous(), "Ybatch")
   self:check(X_starbatch:contiguous(), "X_star")

   print(string.format([[
CMFA Training batch %d
-----------------]], batchIdx))

   -- self.conditional.Xm:index(self.Xm_N, 3, batchperm)
   self:reset()
   self.conditional.Xm:copy(X_starbatch:repeatTensor(s, 1, 1))

   -- self.hidden.Zm:index(self.Zm_N, 3, batchperm)
   -- self.hidden.Qs:index(self.Qs_N, 1, batchperm)

   self:rho(batchIdx)
   print(string.format("rho = %f\n", self:rho()))
   print(string.format("Prev Qs responsibility = %s", self.hidden.Qs:sum(1)))

   for epoch = 1, epochs do
      -- for citr = 1, 5 do
      for citr = 1, 10 do
         self:infer("QZ", pause, debug, Ybatch)
         self:infer("QX", pause, debug, Ybatch, X_starbatch)
      end

      for citr = 1, 10 do
         self:infer("QL", pause, debug, Ybatch)
         self:infer("QG", pause, debug, Ybatch)
         self:infer("Qnu", pause, debug)
         self:infer("Qomega", pause, debug)
      end
      -- end

      self:infer("Qs", pause, debug, Ybatch)
      self:infer("Qpi", pause, debug)

      -- print(self.hidden.Qs:sum(1))
      -- print("\n")

      if epoch % 2 == 0 then
         for ctr = 1, 5 do
            self:infer("PsiI", pause, debug, Ybatch)
            self:infer("E_starI", pause, debug, X_starbatch)
         end
      end
      xlua.progress(epoch, epochs)
   end
   xlua.progress(epochs, epochs)

   self:infer("Qs", pause, debug, Ybatch, true)
   print(string.format("sizeOfS %s", self._sizeofS))

   local F, dF = self:calcF(debug, Ybatch, X_starbatch)
   print(string.format("F = %f \t dF = %f", F, dF))

   -- self.Xm_N:indexCopy(3, batchperm, self.conditional.Xm)
   self.Zm_N:indexCopy(3, batchperm, self.hidden.Zm)
   self.Qs_N:indexCopy(1, batchperm, self.hidden.Qs)

   return F
end

-- Trains the CMFA model using Stochastic
-- variational updates in batches
function CMFA:train(Y, X_star, epochs)
   local n, s, p, k, f, N = self:_setandgetDims()

   assert(self:_checkDimensions(Y, p, N))
   assert(self:_checkDimensions(X_star, f, N))

   self:check(Y, "Y")
   self:check(X_star, "X_star")

   -- create randomly permuted dataset for each epoch

   local Ybatch = torch.zeros(p, n)
   local X_starbatch = torch.zeros(f, n)

   -- local Xm_N = torch.randn(f, n):repeatTensor(s, 1, 1)
   local Xm_N = X_star:repeatTensor(s, 1, 1) --
   local Zm_N = torch.randn(s, k, N)
   local Qs_N = torch.ones(N, s) / s

   self:check(Xm_N, "Xm_N")
   self:check(Zm_N, "Zm_N")
   self:check(Qs_N, "Qs_N")

   local pause = true
   local debug = false

   self.old = {}

   for epoch = 1, epochs do
      local permutation = torch.randperm(N):long()
      local batches = permutation:split(n)

      -- every epoch consist of multiple iteration
      -- to allow convergence
      for itr = 1, 3  do
         print(string.format([[
--------------------------------------------
Epoch %d's Convergence iteration number = %d
--------------------------------------------]], epoch, itr))

         local itrStart = os.clock()
         self._sizeofS = torch.zeros(s)

         -- for batchIdx = 1, (N / n) do
         for batchIdx, batch in ipairs(batches) do
            print(string.format([[
Training batch %d
-----------------]], batchIdx))

            -- prepare batches
            Ybatch:index(Y, 2, batch)
            X_starbatch:index(X_star, 2, batch)
            self.conditional.Xm:index(Xm_N, 3, batch)
            self.hidden.Zm:index(Zm_N, 3, batch)
            self.hidden.Qs:index(Qs_N, 1, batch)

            self:rho(batchIdx)
            print(string.format("rho = %f\n", self:rho()))

            pause = false
            debug = false

            if debug then
               self.old.Xm = self.conditional.Xm:clone()
               self.old.Zm = self.hidden.Zm:clone()
               self.old.Zcov = self.hidden.Zcov:clone()
               self.old.Lm = self.factorLoading.Lm:clone()
               self.old.Lcov = self.factorLoading.Lcov:clone()
               self.old.Gm = self.factorLoading.Gm:clone()
               self.old.Gcov = self.factorLoading.Gcov:clone()
               self.old.E_starI = self.hyper.E_starI:clone()
            end

            for citr = 1, 20 do
               -- for citr = 1, 10 do
               for citr = 1, 5 do
                  self:infer("QZ", pause, debug, Ybatch)
                  self:infer("QX", pause, debug, Ybatch, X_starbatch)
               end

               for citr = 1, 5 do
                  self:infer("QL", pause, debug, Ybatch)
                  self:infer("QG", pause, debug, Ybatch)
                  self:infer("Qnu", pause, debug)
                  self:infer("Qomega", pause, debug)
               end
               -- end

               self:infer("Qs", pause, debug, Ybatch)
               self:infer("Qpi", pause, debug)

               if citr % 3 == 0 then
                  for ctr = 1, 10 do
                     self:infer("PsiI", pause, debug, Ybatch)
                     self:infer("E_starI", pause, debug, X_starbatch)
                  end
               end
            end

            if debug then
               self:pr("Zm", "hidden", pause)
               self:pr("Zcov", "hidden", pause)
               self:pr("Xm", "conditional", pause)
               self:pr("Xcov", "conditional", pause)
               self:pr("Lm", "factorLoading", pause)
               self:pr("Lcov", "factorLoading", pause)
               self:pr("Gm", "factorLoading", pause)
               self:pr("Gcov", "factorLoading", pause)
               self:pr("E_starI", "hyper", pause)
            end

            pause = false
            debug = false

            self:infer("Qs", pause, debug, Ybatch, true)
            print(string.format("sizeOfS %s", self._sizeofS))

            local F, dF = self:calcF(debug, Ybatch, X_starbatch)
            print(string.format("F = %f \t dF = %f", F, dF))

            if pause then io.read() end

            Xm_N:indexCopy(3, batch, self.conditional.Xm)
            Zm_N:indexCopy(3, batch, self.hidden.Zm)
            Qs_N:indexCopy(1, batch, self.hidden.Qs)
         end --  batch end

         print(string.format("Convergence iteration finished in = %f", os.clock() - itrStart))
      end -- convergence iteration end

      if not self:doremoval() then self:dobirth() end -- either perform removal
                                                      -- or birth
   end -- epoch end

   return self.F, self.factorLoading.Gm, self.factorLoading.Gcov, self.factorLoading.Lm, self.factorLoading.Lcov, Zm_N, Xm_N, Qs_N
end

--
function CMFA:infer(tr, pause, ...)
   local c = os.clock()
   -- print(string.format("\n== Infer %s ==", tr))
   self["infer"..tr](self, ...)
   -- print(string.format("%s\t= %f", tr, os.clock() - c))
   -- print("------------------------------------------")
   if pause then io.read() end
end

function CMFA:pr(tr, ns, pause)
   print(tr .. " Prev")
   print(self.old[tr])

   print(tr .. " New")
   print(self[ns][tr])

   if pause then io.read() end
end
return CMFA