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
   local Xm_N = torch.Tensor(s, f, N)

   for epoch = 1, epochs do
      local permutation = torch.randperm(N):long()
      local batches = permutation:split(n)

      -- permute
      Xm_N = Xm_N:index(3, permutation)

      -- every epoch consist of multiple iteration
      -- to allow convergence
      for itr = 1, 10  do
         -- for each batch
         print(string.format("Convergence iteration number = %d", itr))

         for batchIdx, batch in ipairs(batches) do
            print(string.format("Training batch %d", batchIdx))
            -- prepare batches
            Ybatch:copy(Y:index(2, batch))
            X_starbatch:copy(X_star:index(2, batch))
            self.conditional.Xm = Xm_N:narrow(3, (batchIdx - 1) * n + 1, n)

            -- traning initialization
            self:inferQZ(Ybatch)
            self:inferQX(Ybatch, X_starbatch)

            if itr == itrs then self:inferQs(Ybatch, true) -- in last iteration
            else self:inferQs(Ybatch) end                  -- always calculate sizeofS

            self:inferQnu()
            self:inferQomega()

            -- learn
            self:inferQZ(Ybatch)
            self:inferQX(Ybatch, X_starbatch)
            self:inferQL(Ybatch)
            self:inferQG(Ybatch)
            self:inferQs(Ybatch)
            self:inferQpi()
            self:inferQnu()
            self:inferQomega()
            self:inferPsiI(Ybatch)
            self:inferE_starI(X_starbatch)

            self:calcF()
         end
      end

      if not self:doremoval() then self:dobirth() end -- either perform removal
                                                      -- or birth
   end

end

return CMFA