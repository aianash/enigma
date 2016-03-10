--------------------------------------------------
--[[ CMFA ]]--
-- Conditonal Mixture of Factor Analysers
--------------------------------------------------
local CMFA = {}

local parent = VBMFA:_factory()
setmetatable(CMFA, parent)

function CMFA:new( ... )
   local o = {}
   setmetatable(o, self)
   self.__index = self
   o:__init(...)
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

   local Ybatch = torch.zeros(p, n, torch.getdefaulttensortype())
   local X_starbatch = torch.zeros(f, n, torch.getdefaulttensortype())

   for epoch in 1, epochs do
      -- train in the epoch
      local indices = torch.randperm(N):long():split(n)
      for t, i in ipairs(indices) do
         
      end

      -- decide whether to perform birth and death in the next
      -- epoch
   end   

end