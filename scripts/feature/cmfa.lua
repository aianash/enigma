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

   -- Xm is treated especially,
   -- to enable carry forward accross epoch
   -- initialize it with X_star ie the input
   local Xm_N = X_star:repeatTensor(s, 1, 1) --torch.randn(f, n):repeatTensor(s, 1, 1)
   local Zm_N = torch.randn(s, k, N)
   local Qs_N = torch.ones(N, s) / s

   local bStart

   local pause = true
   local debug = false

   self.old = {}

   for epoch = 1, epochs do
      -- local permutation = torch.randperm(N):long()
      -- local batches = permutation:split(n)

      print(string.format("In epoch = %d", epoch))

      -- every epoch consist of multiple iteration
      -- to allow convergence
      for itr = 1, 3  do
         -- for each batch
         print(string.format([[
---------------------------------
Epoch %d's Convergence iteration number = %d
---------------------------------]], epoch, itr))
         -- print(Xcov_N[{ {}, 1, {}, {} }])

         -- local Xm_N_P = Xm_N:index(3, permutation)
         -- local Xcov_N_P = Xcov_N:index(2, permutation)
         -- local Zm_N_P = Zm_N:index(3, permutation)
         -- local Qs_N_P = Qs_N:index(1, permutation)
         -- local E_starI_N_P = E_starI_N:index(1, permutation)

         -- for batchIdx, batch in ipairs(batches) do
         self._sizeofS = torch.zeros(s)

         for batchIdx = 1, (N / n) do
            print(string.format([[
Training batch %d
-----------------]], batchIdx))

            collectgarbage()
            bStart = os.time()

            -- prepare batches
            Ybatch:copy(Y:narrow(2, (batchIdx - 1) * n + 1, n)) -- :index(2, batch))
            X_starbatch:copy(X_star:narrow(2, (batchIdx - 1) * n + 1, n)) -- :copy(X_star:index(2, batch))

            self.conditional.Xm = Xm_N:narrow(3, (batchIdx - 1) * n + 1, n)
            self.hidden.Zm = Zm_N:narrow(3, (batchIdx - 1) * n + 1, n)
            self.hidden.Qs = Qs_N:narrow(1, (batchIdx - 1) * n + 1, n)
            -- self.hyper.E_starI = E_starI_N:narrow(1, (batchIdx - 1) * n + 1, n)
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
            end

            -- Fixed point convergence
            for citr = 1, 5 do
               for citr = 1, 5 do
                  for citr2 = 1, 10 do
                     self:infer("QZ", pause, debug, Ybatch)
                     self:infer("QX", pause, debug, Ybatch, X_starbatch)
                  end

                  for citr3 = 1, 10 do
                     self:infer("QL", pause, debug, Ybatch)
                     self:infer("QG", pause, debug, Ybatch)
                     self:infer("Qnu", pause, debug)
                     self:infer("Qomega", pause, debug)
                  end
               end

               -- for citr = 1, 10 do
               -- end
               for citr = 1, 5 do
                  self:infer("Qpi", pause, debug)
               end

               for citr = 1, 5 do
                  self:infer("Qs", pause, debug, Ybatch)
               end

               if citr % 3 == 0 then
                  for ctr = 1, 10 do
                     self:infer("PsiI", pause, debug, Ybatch)
                  end

                  -- for ctr = 1, 10 do
                  --    self:infer("E_starI", pause, debug, X_starbatch)
                  -- end
               end
            end

            -- self:infer("Qs", pause, debug, Ybatch, true)

            self._sizeofS:add(self.hidden.Qs:sum(1))
            print(string.format("sizeOfS %s", self._sizeofS))

            if debug then
               self:pr("Zm", "hidden", pause)
               self:pr("Zcov", "hidden", pause)
               self:pr("Xm", "conditional", pause)
               self:pr("Xcov", "conditional", pause)
               self:pr("Lm", "factorLoading", pause)
               self:pr("Lcov", "factorLoading", pause)
               self:pr("Gm", "factorLoading", pause)
               self:pr("Gcov", "factorLoading", pause)
            end

            -- [[ Hyper parameter optimization ]] --
            pause = false
            debug = false

            if debug then self.old.E_starI = self.hyper.E_starI:clone() end
            -- for ctr = 1, 10 do
            --    self:infer("E_starI", pause, debug, X_starbatch)
            -- end
            if debug then self:pr("E_starI", "hyper", pause) end

            local F, dF = self:calcF(debug, Ybatch, X_starbatch)

            print(self.factorLoading.Gm)
            print(string.format("F = %f \t dF = %f", F, dF))

            if pause then io.read() end
         end

         -- Xm_N:indexCopy(3, permutation, Xm_N_P)
         -- Xcov_N:indexCopy(2, permutation, Xcov_N_P)
         -- Zm_N:indexCopy(3, permutation, Zm_N_P)
         -- Qs_N:indexCopy(1, permutation, Qs_N_P)
         -- E_starI_N:indexCopy(1, permutation, E_starI_N_P)

      end

      if not self:doremoval() then self:dobirth() end -- either perform removal
                                                      -- or birth
   end

   return self.factorLoading.Gm, self.factorLoading.Gcov, self.factorLoading.Lm, self.factorLoading.Lcov
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