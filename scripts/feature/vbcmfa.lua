local VBCMFA = {}

----------------------
--[[ MFA ]]--
-- Implements the Variational Bayesian
-- Mixture of Factor Analyzer
-- y = Lz + Gx + e | s or Y = LZ + GX + e | S
-- y  - p
-- Y  - p x n
-- L  - s x p x k
-- z  - s x k
-- Z  - s x k x n
-- G  - s x p x f
-- x  - s x f
-- X  - x x f x n
-- S  - n x s
--
-- Priors
-- p(L | 0, nu)
-- p(nu | a_star, b_star)
-- p(omega | alpha_star, beta_star)
-- p(s | pi)
-- p(pi | phi_star)
-- p(e | 0, Psi)
-- 
-- Posteriors
-- q(L | Lm, Lcov)
-- q(Z | Zm, Zcov, S)
-- q(nu | a, b)
--
-- q(G | Gm, Gcov)
-- q(X | Xm, Xcov, S)
-- q(omega | alpha, beta)
-- 
-- q(s)
-- q(pi | phim)
-- 
-- Hidden variables
-- Zm   - s x k x n            (hidden params)
-- Zcov - s x k x k            (hidden params)
-- 
-- Xm   - s x f x n
-- Xcov - s x f x f
--
-- Qs   - n x s
-- phim - s
--
-- Factor Loading parameters
-- Lm    - s x p x k             (each s component, each row p, k dimension of mean of Lambda and 1 for mean vector)
-- Lcov  - s x p x k x k   (each s component, each row, p, kxk - dimensional cov matrix)
-- a     - 1
-- b     - s x k
--
-- Gm    - s x p x f
-- Gcov  - s x p x f x f
-- alpha - 1
-- beta  - s x f
--
-- Hyper parameters
-- a_star     - 1                        (hyper parameters for priori
-- b_star     - 1                        on 
--
-- alpha_star - 1
-- beta_star  - 1
-- 
-- phi_star   - 1                        (a number)
--
-- PsiI       - p                        (a diagonal matrix)
----------------------
function VBCMFA:new(...)
   local o = {}
   setmetatable(o, self)
   self.__index = self
   o:__init(...)
   return o
end

--
function VBCMFA:__init(cfg)
   local n, s, p, k, f = self:_setandgetDims(cfg)

   self.hidden = {
      Zm = torch.Tensor(s, k, n),
      Zcov = torch.Tensor(s, k, k):zeros(),

      Xm = torch.Tensor(s, f, n),
      Xcov = torch.Tensor(s, f, f):zeros(),

      Qs = torch.Tensor(n, s),
      phim = torch.Tensor(s),

      getX = function(self)
         return self.Xm, self.Xconv
      end,

      getZ = function(self)
         return self.Zm, self.Zcov
      end,

      getS = function(self)
         return self.Qs, self.phim
      end
   }

   self.factorLoading = {
      Lm = torch.Tensor(s, p, k),
      Lcov = torch.Tensor(s, p, k, k),

      a = 1,
      b = torch.Tensor(s, p),

      Gm = torch.Tensor(s, p, f),
      Gcov = torch.Tensor(s, p, f, f)

      alpha = 1,
      beta = torch.Tensor(s, p),

      getL = function(self)
         return self.Lm, self.Lcov, self.a, self.b
      end,

      getG = function(self)
         return self.Gm, self.Gcov, self.alpha, self.beta
      end
   }

   self.hyper = {
      a_star = 1,
      b_star = 1,

      alpha_star = 1,
      beta_star = 1, 

      E_starI = torch.Tensor(f, f),

      phi_star = 1,

      PsiI = torch.Tensor(p),

      get = function(self)
         return self.a_star, self.b_star, self.alpha_star, self.beta_star, self.E_starI, self.phi_star, self.PsiI
      end
   }

   self:initparams()
end

--
function VBCMFA:_setandgetDims(cfg)
   if cfg then
      self.n = cfg.batchSize
      self.s = cfg.numComponents
      self.p = cfg.outputVectorSize
      self.k = cfg.factorVectorSize
      self.f = cfg.inputVectorSize
   end
   return self.n, self.s, self.p, self.k, self.f
end

------------
--[[ Lz ]]--
------------

--
function VBCMFA:inferQL(Y) -- p x n
   local n, s, p, k, f = self:_setandgetDims()   
   local Lm, Lcov, a, b = self.factorLoading:getL()
   local Gm, Gcov = self.factorLoading:getG()
   local Zm, Zcov = self.hidden:getZ()
   local Xm = self.hidden:getX()
   local Qs = self.hidden:getS()
   
   for t = 1, s do
      local Zmt = Zm[t] -- k x n
      local Qst = Qs[{ {}, t }]
      local kQst = Qst:repeatTensor(k, 1) -- k x n
      local ZmtkQst = torch.cmul(Zmt, kQst) -- k x n

      local nY = Y - Gm[t] * Xm[t] -- p x n
      local PsiInYZmtkQstT = torch.diag(PsiI) * nY * ZmtkQst:t() -- p x k

      local QstEzzT = Zcov[t] * torch.sum(Qst) + Zmt * ZmtkQst:t() -- k x k
      local a_bt = torch.div(b[t], a):cinv() -- k

      for q = 1, p do
         torch.inverse(Lcov[t][q], torch.diag(a_bt) + PsiI[q] * QstEzzT) -- inv{ k x k + 1 * k x k }
         Lm[t][q]:mv(Lcov[t][q], PsiIYnZmtkQstT[q]) -- k x k * k x 1 = k x 1 
      end
   end
end

--
function VBCMFA:inferQZ(Y)
   local n, s, p, k, f = self:_setandgetDims()
   local Lm, Lcov, a, b = self.factorLoading:getL()
   local Gm = self.factorLoading:getG()
   local Zm, Zcov = self.hidden:getZ()
   local Xm = self.hidden:getX()


   for t = 1, s do
      local Lmt = Lm[t]
      local Lcovt = Lcov[t]
      local Zcovt = Zcov[t]

      local LmtTPsiI = Lmt:t() * PsiI -- k x p
      
      local nY = Y - Gm[t] * Xm[t] -- p x n
      local LmtTPsiInY = LmtTPsiI * nY -- k x n

      local ELTPsiIL = torch.reshape(torch.reshape(Lcovt, p, k * k):t() * PsiI, k, k)
                           + LmtTPsiI * Lmt -- k x k

      torch.inverse(Zcovt, torch.eye(k) + ELTPsiIL) -- k x k
      Zm[t]:mm(Zcovt, LmtPsiInY) -- k x k * k x n
   end
end

--
function VBCMFA:inferQnu( ... )
   local n, s, p, k, f = self:_setandgetDims()
   local Lm, Lcov, _, b = self.factorLoading:getL()
   local a_star, b_star = self.hyper.a_star, self.hyper.b_star
   
   self.factorLoading.a = a_star + p / 2
   for t = 1, s do
      local Lmt = Lm[t] -- p x k
      local Lcovt = Lcov[t] -- p x k x k
      local EL_sqr = torch.diag(torch.sum(Lcovt, 1)[1]) + torch.sum(torch.pow(Lmt, 2), 1)[1] -- k
      b[t] = bstr * torch.ones(k) + 0.5 * EL_sqr
   end
   local n, s, p, k, f = self:_setandgetDims()
   local Lm, Lcov, _, b = self.factorLoading:getL()
   local a_star, b_star = self.hyper.a_star, self.hyper.b_star
   
   self.factorLoading.a = a_star + p / 2
   for t = 1, s do
      local Lmt = Lm[t] -- p x k
      local Lcovt = Lcov[t] -- p x k x k
      local EL_sqr = torch.diag(torch.sum(Lcovt, 1)[1]) + torch.sum(torch.pow(Lmt, 2), 1)[1] -- k
      b[t] = beta_star * torch.ones(k) + 0.5 * EL_sqr
   end


------------
--[[ Gx ]]--
------------

--
function VBCMFA:inferQG( ... )
   local n, s, p, k, f = self:_setandgetDims()   
   local Gm, Gcov, alpha, beta = self.factorLoading:getG()
   local Lm, Lcov = self.factorLoading:getL()
   local Xm, Xcov = self.hidden:getZ()
   local Zm = self.hidden:getX()
   local Qs = self.hidden:getS()
   
   for t = 1, s do
      local Xmt = Xm[t] -- f x n
      local Qst = Qs[{ {}, t }]
      local fQst = Qst:repeatTensor(f, 1) -- f x n
      local XmtfQst = torch.cmul(Xmt, fQst) -- f x n

      local nY = Y - Lm[t] * Zm[t] -- p x n
      local PsiInYXmtfQstT = torch.diag(PsiI) * nY * XmtfQst:t() -- p x f

      local QstExxT = Xcov[t] * torch.sum(Qst) + Xmt * XmtfQst:t() -- f x f
      local alpha_betat = torch.div(beta[t], alpha):cinv() -- f

      for q = 1, p do
         torch.inverse(Gcov[t][q], torch.diag(alpha_betat) + PsiI[q] * QstExxT) -- inv{ k x k + 1 * k x k }
         Gm[t][q]:mv(Gcov[t][q], PsiIYnXmtfQstT[q]) -- k x k * k x 1 = k x 1 
      end
   end
end

--
function VBCMFA:inferQX( ... )
   local n, s, p, k, f = self:_setandgetDims()
   local Gm, Gcov = self.factorLoading:getG()
   local Xm, Xcov = self.hidden:getX()
   local Lm = self.factorLoading:getL()
   local Zm = self.hidden:getZ()
   local E_starI = self.hyper.E_starI

   for t = 1, s do
      local Gmt = Gm[t]
      local Gcovt = Gcov[t]
      local Xcovt = Xcov[t]

      local GmtTPsiI = Gmt:t() * PsiI -- f x p
      
      local nY = Y - Lm[t] * Zm[t] -- p x n
      local GmtTPsiInY = GmtTPsiI * nY -- f x n

      local EGTPsiIG = torch.reshape(torch.reshape(Gcovt, p, f * f):t() * PsiI, f, f)
                           + GmtTPsiI * Gmt -- f x f

      torch.inverse(Xcovt, E_starI + EGTPsiIG) -- f x f
      Xm[t]:mm(Xcovt, E_starI * X_star + GmtPsiInY)
      --       f x f * { f x f * f x n + f x n }
   end
end

--
function VBCMFA:inferQomega( ... )
   local n, s, p, k, f = self:_setandgetDims()
   local Gm, Gcov, _, beta = self.factorLoading:getG()
   local alpha_star, beta_star = self.hyper.alpha_star, self.hyper.beta_star
   
   self.factorLoading.alpha = alpha_star + p / 2
   for t = 1, s do
      local Gmt = Gm[t] -- p x f
      local Gcovt = Gcov[t] -- p x f x f
      local EL_sqr = torch.diag(torch.sum(Gcovt, 1)[1]) + torch.sum(torch.pow(Gmt, 2), 1)[1] -- f
      beta[t] = beta_star * torch.ones(f) + 0.5 * EL_sqr
   end
end

-----------
--[[ S ]]--
-----------

--
function VBCMFA:inferQs( ... )
   
end

--
function VBCMFA:inferQpi( ... )
   local n, s, p, k, f = self:_setandgetDims()
   local Qs, phim = self.hidden:getS()
   local phi_starm = self.hyper.phi_star / s * torch.ones(s)

   phim:add(phi_star, torch.sum(Qs, 1)[1])
end

--------------------------
--[[ Hyper parameters ]]--
--------------------------

--
function VBCMFA:inferPsiI( ... )
   
end

--
function VBCMFA:inferab( ... )
   
end

--
function VBCMFA:inferalphabeta( ... )
   -- body
end

--
function VBCMFA:inferPhi( ... )
   -- body
end

--
function VBCMFA:inferE_starI( ... )
   -- body
end
