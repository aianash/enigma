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
   local Xm_N = X_star:repeatTensor(s, 1, 1)

   local bStart

   for epoch = 1, epochs do
      local permutation = torch.randperm(N):long()
      local batches = permutation:split(n)

      print(string.format("In epoch = ", epoch))

      -- permute
      Xm_N = Xm_N:index(3, permutation)

      -- every epoch consist of multiple iteration
      -- to allow convergence
      for itr = 1, 3  do
         -- for each batch
         print(string.format([[
---------------------------------
Convergence iteration number = %d
---------------------------------]], itr))

         for batchIdx, batch in ipairs(batches) do
            print(string.format([[
Training batch %d
-----------------]], batchIdx))
            collectgarbage()
            bStart = os.time()
            -- prepare batches
            Ybatch:copy(Y:index(2, batch))
            X_starbatch:copy(X_star:index(2, batch))
            self.conditional.Xm = Xm_N:narrow(3, (batchIdx - 1) * n + 1, n)

            local t = os.clock()

            -- traning initialization
            self:inferQZ(Ybatch)
            print(string.format("QZ\t= %f", os.clock() - t))
            t = os.clock()

            self:inferQX(Ybatch, X_starbatch)
            print(string.format("QX\t= %f", os.clock() - t))
            t = os.clock()

            if itr == itrs then self:inferQs(Ybatch, true) -- in last iteration
            else self:inferQs(Ybatch) end                  -- always calculate sizeofS
            print(string.format("Qs\t= %f", os.clock() - t))
            t = os.clock()

            self:inferQnu()
            self:inferQomega()


            -- learn
            self:inferQZ(Ybatch)
            print(string.format("QZ\t= %f", os.clock() - t))
            t = os.clock()

            self:inferQX(Ybatch, X_starbatch)
            print(string.format("QX\t= %f", os.clock() - t))
            t = os.clock()

            collectgarbage()

            self:inferQL(Ybatch)
            print(string.format("QL\t= %f", os.clock() - t))
            t = os.clock()

            self:inferQG(Ybatch)
            print(string.format("QG\t= %f", os.clock() - t))
            t = os.clock()

            -- collectgarbage()

            self:inferQs(Ybatch)
            print(string.format("Qs\t= %f", os.clock() - t))
            t = os.clock()

            self:inferQpi()
            print(string.format("Qpi\t= %f", os.clock() - t))
            t = os.clock()

            self:inferQnu()
            print(string.format("Qnu\t= %f", os.clock() - t))
            t = os.clock()

            self:inferQomega()
            print(string.format("Qomega\t= %f", os.clock() - t))
            t = os.clock()

            self:inferPsiI(Ybatch)
            print(string.format("PsiI\t= %f", os.clock() - t))
            t = os.clock()

            self:inferE_starI(X_starbatch)
            print(string.format("E_starI\t= %f", os.clock() - t))
            t = os.clock()

            self:calcF()
            print(string.format("in\t= %d secs\n", os.time() - bStart))
         end
      end

      if not self:doremoval() then self:dobirth() end -- either perform removal
                                                      -- or birth
   end

end

return CMFA