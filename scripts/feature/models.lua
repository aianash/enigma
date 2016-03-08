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
-- Priors
-- p(L | 0, v)
-- p(nu | astr, bstr)
-- p(u | ustr, vstr)
-- p(s | pi)
-- p(pi | alphastr)
-- p(e | 0, Psi)
-- 
-- Posteriors
-- q(_L | _Lm, _Lcov)
-- q(_X | _Xm, _Xcov)
-- q(s)
-- q(pi | am)
--
-- Hidden variables
-- _Xm   - s x (k + 1) x n             (hidden params)
-- _Xcov - s x (k + 1) x (k + 1)       (hidden params)
-- Qs    - n x s
-- am    - s
--
-- Factor Loading parameters
-- _Lm   - s x p x (k + 1)             (each s component, each row p, k dimension of mean of Lambda and 1 for mean vector)
-- _Lcov - s x p x (k + 1) x (k + 1)   (each s component, each row, p, kxk - dimensional cov matrix)
-- a     - 1
-- b     - s x k
--
-- Hyper parameters
-- ustr    - p
-- vstr    - p                         (a diagonal covariance matrix) 
-- 
-- astr     - 1                        (hyper parameters for priori
-- bstr     - 1                        on 
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
         Qs = torch.Tensor(n, s),
         am = torch.Tensor(s),

         get = function(self)
            return self._Xm, self._Xconv, self.Qs, self.am
         end
      }

      self.factorLoading = {
         _Lm = torch.Tensor(s, p, k + 1),
         _Lcov = torch.Tensor(s, p, k + 1, k + 1),
         a = 1,
         b = torch.Tensor(s, p),

         get = function(self)
            return self._Lm, self._Lcov, self.a, self.b
         end
      }

      self.hyper = {
         ustr = torch.Tensor(p),
         vstr = torch.Tensor(p),

         astr = 1,
         bstr = 1,

         alphastr = 1,

         PsiI = torch.Tensor(p),

         get = function(self)
            return self.ustr, self.vstr, self.astr, self.bstr, self.alphastr, self.PsiI
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
      local _Xm, _Xcov, Qs = self.hidden:get()
      local _Lm, _Lcov = self.factorLoading:get()

      local PsiI = self.hidden.PsiI

      for t = 1, s do
         local Lmt = _Lm[t]:narrow(2, 2, k) -- p x k 
         local Lcovt = _Lcov[t][{ {}, {2, k + 1}, {2, k + 1} }] -- p x k x k
         local Xcovt = _Xcov[t]:sub(2, k + 1, 2, k + 1) -- k x k

         local LmtTPsiI = Lmt:t() * torch.diag(PsiI) -- k x p

         local E_qL = torch.reshape(torch.reshape(Lcovt, p, k * k):t() * PsiI, k, k)
                        + LmtTPsiI * Lmt -- k x k
         torch.inverse(Xcovt, torch.eye(k) + E_qL) -- k x k

         local Xmt = _Xm[t]:narrow(1, 2, k)  -- assumed _Xm is initialized with ones everywhere
         local umt = _Lmt[t]:select(2, 1)
         E_qL = LmtTPsiI * (Y - umt * torch.ones(1, n))
         Xmt:mm(_Xcovt, E_qL)
      end

   end

   --
   function VBMFA:inferQL(Y, targetY)
      local n, s, p, k = self:_setandgetDims()
      local _Xm, _Xcov, Qs = self.hidden:get()
      local _Lm, _Lcov, a, b = self.factorLoading:get()
      local ustr, vstr, _, _, alphastr, PsiI = self.hyper:get()


      for t = 1, s do
         local _Xmt = Xm[t] -- (k + 1) x n
         local Qst = Qs[{{}, t}] -- n x 1
         local kQst = Qst:repeatTensor(k + 1, 1) -- (k + 1) x n
         local _XmtQst = torch.cmul(_Xmt, kQst) -- (k + 1) x n

         local PsiIY_XmtQst = diag(PsiI) * Y * _XmtQst:t() -- p x (k + 1)
         local ustr_expnd = torch.cat(ustr, torch.zeros(p, k), 2) -- p x (k + 1)

         local abt = torch.div(b[t], a):cinv():repeatTensor(p, 1) -- p x k
         local vstr_abt = torch.cat(vstr, abt, 2) -- p x (k + 1)

         local E = _Xcov[t] * torch.sum(Qst) + _Xmt * _XmtQst:t() -- (k + 1) x (k + 1)

         for q = 1:p do
            torch.inverse(_Lcov[t][q], torch.diag(vstr_abt[q]) + E * PsiI[q]) -- inv{ (k + 1) x (k + 1) + (k + 1) x (k + 1) }
            _Lm[t][q]:mm(PsiIY_XmtQst[q] + ustr_expnd[q] * torch.diag(vstr_abt[q]), _Lcov[t][q])
                     -- { p x (k + 1)     + (p x (k + 1)  * ((k + 1) x (k + 1))}    * ((k + 1) x (k + 1))
         end
      end
   end

   --
   function VBMFA:inferQs(Y, targetY)
      local n, s, p, k = self:_setandgetDims()
      local _Xm, _Xcov, Qs = self.hidden:get()
      local _Lm, _Lcov, a, b = self.factorLoading:get()
      local ustr, vstr, _, _, alphastr, PsiI = self.hyper:get()

      local logQs = torch.Tensor(n, s)

      for t = 1, s do

      end
   end

   --
   function VBMFA:inferQnu()
      local n, s, p, k = self:_setandgetDims()
      local _Lm, _Lcov, _, b = self.factorLoading:get()
      local _, _, astr, bstr, _, _ = self.hyper:get()
      
      self.a = astr + p / 2
      for t = 1, s do
         local Lmt = _Lm[t][{ {}, {2, k + 1} }] -- p x k
         local Lcovt = _Lcov[t][{ {}, {2, k + 1}, {2, k + 1} }] -- p x k x k
         local E = torch.diag(torch.sum(Lcovt, 1)[1]) + torch.sum(torch.pow(Lmt, 2), 1)[1] -- k
         b[t] = bstr * torch.ones(k) + 0.5 * E -- k
      end
   end

   --
   function VBMFA:inferQpi()
      local n, s, p, k = self:_setandgetDims()
      local _, _, Qs, am = self.hidden:get()
      local alphastrm = alphastr / s * torch.ones(s)

      am:add(alphastrm, torch.sum(Qs, 1)[1])
   end

   --
   function VBMFA:inferPsiI(Y, targetY)
      local n, s, p, k = self:_setandgetDims()
      local _Xm, _Xcov, Qs = self.hidden:get()
      local _Lm, _Lcov, a, b = self.factorLoading:get()

      local psi = torch.zeros(p)

      for t = 1, s do
         local kQst = Qst:repeatTensor(k + 1, 1) -- (k + 1) x n
         local _XmtQst = torch.cmul(_Xmt, kQst) -- (k + 1) x n

         local E = _Xcov[t] * torch.sum(Qst) + _Xmt * _XmtQst:t() -- (k + 1) x (k + 1)
         local pQst = Qst:repeatTensor(p, 1) -- p x n
         local pQstY = torch.cmul(Y, pQst) -- p x n 
         psi = psi + pQstY * (Y - 2 * _Lm[t] * _Xm[t]):t() + _Lm[t] * E * _Lm[t]:t()
         for q = 1, p do
            psi[q] = psi[q] + torch.trace(_Lcov[t][q] * E)
         end
      end

      torch.div(PsiI, psi, n)
      PsiI:cinv()
   end

   --
   function VBMFA:inferuvstr()
      local n, s, p, k = self:_setandgetDims()
      local _Lm, _Lcov, a, b = self.factorLoading:get()
      local ustr, vstr, _, _, alphastr, PsiI = self.hyper:get()

      ustr:mean(_Lm[{ {}, {}, 1 }], 1):resize(p)
      local vstrI = torch.pow(_Lm[{ {}, {}, 1 }], 2):sum(1):resize(p)
                     - s * torch.pow(ustr, 2)
                     + 
      torch.div(vstr, vstrI, s)
      vstr:cinv()
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