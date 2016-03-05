local models = {}

----------------------
--[[ MFA ]]--
-- Implements the Variational Bayesian
-- Mixture of Factor Analyzer
-- y = Lx + u + e | s or y = _L_x + e | s or Y = _L_X + e | S
-- in second case L and u are combined, and 1 is added to x
-- y  - p
-- Y  - p x n
-- u  - s x p
-- L  - s x p x k
-- x  - s x k
-- X  - s x k x n
-- S  - n x s
-- _L - s x p x (k + 1)                (combined Factor loading matrix - this is actually used)
-- _x - s x (k + 1)                    (x with 1 concatenated)
-- _X - s x (k + 1) x n
--
-- Hidden variables
-- _Xm   - s x (k + 1) x n             (hidden params)
-- _Xcov - s x (k + 1) x (k + 1)       (hidden params)
-- Qns   - n x s
--
-- Factor Loading parameters
-- _Lm   - s x p x (k + 1)             (each s component, each row p, k dimension of mean of Lambda and 1 for mean vector)
-- _Lcov - s x p x (k + 1) x (k + 1)   (each s component, each row, p, kxk - dimensional cov matrix)
--
-- Hyper parameters
-- mustr    - p
-- nustr    - p                        (a diagonal covariance matrix) 
-- 
-- astr     - 1
-- bstr     - 1
--
-- alphastr - 1                        (a number)
--
-- PsiI     - p                        (a diagonal matrix)
----------------------
local VBMFA = {}
do
   function VBMFA:new(...)
      local o = {}
      setmetatable(o, self)
      self.__index = self
      o:__init(...)
      return o
   end

   --
   function VBMFA:__init(cfg)
      local n, s, p, k = self:_setandgetDims(cfg)

      self.hidden = {
         _Xm = torch.Tensor(s, k + 1, n),
         _Xcov = torch.Tensor(s, k + 1, k + 1):zeros(),
         Qns = torch.Tensor(n, s),

         get = function(self)
            return self._Xm, self._Xconv
         end
      }

      self.factorLoading = {
         _Lm = torch.Tensor(s, p, k + 1),
         _Lcov = torch.Tensor(s, p, k + 1, k + 1),

         get = function(self)
            return self._Lm, self._Lcov
         end
      }

      self.hyper = {
         mustr = torch.Tensor(p),
         nustr = torch.Tensor(p),

         astr = 1,
         bstr = 1,

         alphastr = 1,

         PsiI = torch.Tensor(p),

         get = function(self)
            return self.mustr, self.nustr, self.astr, self.bstr, self.alphastr, self.PsiI
         end
      }

      self:initparams()
   end

   function VBMFA:_setandgetDims(cfg)
      if cfg then
         self.n = cfg.batchSize
         self.s = cfg.numComponents
         self.p = cfg.outputVectorSize
         self.k = cfg.factorVectorSize
      end
      return self.n, self.s, self.p, self.k
   end

   function VBMFA:_checkDimensions(tensor, ...)
      local dimensions = {...}
      if #dimensions ~= tensor:nDimension() then
         return false, string.format("Wrong number of dimension = %d, expecting %d", tensor:nDimension(), #dimensions) 
      end

      local res = true
      for idx, size in ipairs(dimensions) do
         res = res and (tensor:size(idx) == size)
         if not res then 
            return false, string.format("wrong size = %d at dimension %d, expecting size = %d", tensor:size(idx), idx, size) 
         end
      end

      return true
   end

   --
   function VBMFA:forward(inputs)
      -- Calculate the nu + Lz + e for the most probable
      -- component based on q(si/pi)
      -- return the result

   end

   --
   function VBMFA:backward(Y, targetY, calcGrad)
      local n, s, p, k = self:_setandgetDims()

      assert(self:_checkDimensions(Y, p, n))
      assert(self:_checkDimensions(targetY, p, n))



      -- update all posterior distributions of the model
      -- using targets as data point
      -- self:inferQX(Y, targetY)

      -- caculate the err based on image similarity
      -- metric between targets[1] and targets[2]
      -- (no calculation of grad)
      -- return the err
   end

   --
   function VBMFA:inferQX(Y, targetY)
      local n, s, p, k = self:_setandgetDims()
      local _Xm, _Xcov, Qns = self.hidden:get()
      local _Lm, _Lcov = self.factorLoading:get()

      local PsiI = self.hidden.PsiI

      for t = 1, s do
         local Lmt = _Lm[t]:narrow(2, 2, k) -- p x k 
         local Lcovt = _Lcov[t][{ {}, {2, k + 1}, {2, k + 1} }] -- p x k x k
         local Xcovt = _Xcov[t]:sub(2, k + 1, 2, k + 1) -- k x k

         local LmtTPsiI = Lmt:t() * torch.diag(PsiI) -- k x p

         local E_qL = torch.reshape(torch.reshape(Lcovt, p, k * k):t() * p, k, k)
                        + LmtTPsiI * Lmt
         torch.inverse(Xcovt, torch.eye(k) + E_qL)

         local Xmt = _Xm[t]:narrow(1, 2, k)  -- assumed _Xm is initialized with ones everywhere
         local umt = _Lmt[t]:select(2, 1)
         E_qL = LmtTPsiI * (Y - umt * torch.ones(1, n))
         Xmt:mm(_Xcovt, E_qL)
      end

   end

   -- probably not like this
   function VBMFA:pretrain()

   end

end


function models.newFeaturemodel( ... )
   
end

-- create a Factor Analysis model for glimpse projection
function models.newMFAmodel(glimpseHeight, glimpseWidth)

end

function models.newImageSimilarityCriterion() end

return models