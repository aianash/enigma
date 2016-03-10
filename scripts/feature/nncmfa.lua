--------------------------------------------------
--[[ Neural Network VBCMFA ]]--
--------------------------------------------------
local NNCMFA = {}

local parent = VBMFA:_factory()
setmetatable(NNCMFA, parent)

function NNCMFA:new( ... )
   local o = {}
   setmetatable(o, self)
   self.__index = self
   o:__init(...)
end

function NNCMFA:__init( ... )
   parent:__init( ... )
end

-- returns Y with the current 
function NNCMFA:forward(Y, X_star, batchIdx)
   local n, s, p, k, f = self:_setandgetDims()   
   
   assert(self:_checkDimensions(Y, p, n))
   assert(self:_checkDimensions(X_star, f, n))

   self:_perpareForBatch(batchIdx)
end

-- calculates the new posterior
function NNCMFA:backward(Y, X_star)
   
end