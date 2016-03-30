local VBCMFA = require 'scripts.feature.vbcmfa'

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
end

-- Trains the CMFA model using Stochastic
-- variational updates in batches
function CMFA:train(Y, X_star, epochs)
   local n, s, p, k, f, N = self:_setandgetDims()

   assert(self:_checkDimensions(Y, p, N))
   assert(self:_checkDimensions(X_star, f, N))

   -- create randomly permuted dataset for each epoch

   local Ybatch = torch.zeros(p, n)
   local X_starbatch = torch.zeros(f, n)

   local Xm_N = X_star:repeatTensor(s, 1, 1) --torch.randn(f, n):repeatTensor(s, 1, 1)
   local Zm_N = torch.randn(s, k, N)
   local Qs_N = torch.ones(N, s) / s

   local bStart

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
---------------------------------
Epoch %d's Convergence iteration number = %d
---------------------------------]], epoch, itr))

         self._sizeofS = torch.zeros(s)

         -- for batchIdx = 1, (N / n) do
         for batchIdx, batch in ipairs(batches) do
            print(string.format([[
Training batch %d
-----------------]], batchIdx))

            bStart = os.time()

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
               for citr = 1, 5 do
                  self:infer("QZ", pause, debug, Ybatch)
                  self:infer("QX", pause, debug, Ybatch, X_starbatch)
               end

               self:infer("Qs", pause, debug, Ybatch)

               for citr = 1, 5 do
                  self:infer("QL", pause, debug, Ybatch)
                  self:infer("QG", pause, debug, Ybatch)
                  self:infer("Qnu", pause, debug)
                  self:infer("Qomega", pause, debug)
               end

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
      end -- convergence iteration end

      if not self:doremoval() then self:dobirth() end -- either perform removal
                                                      -- or birth
   end -- epoch end

   return self.factorLoading.Gm, self.factorLoading.Gcov, self.factorLoading.Lm, self.factorLoading.Lcov, Zm_N, Xm_N, Qs_N
end

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