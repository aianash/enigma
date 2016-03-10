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
-- X  - s x f x n
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
-- Xcov - s x n x f x f
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
-- E_starI    - n x f x f
--
-- phi_star   - 1                        (a number)
--
-- PsiI       - p                        (a diagonal matrix)
----------------------
local VBCMFA = {}

--
function VBMFA:_factory()
   local o = {}
   setmetatable(o, self)
   self.__index = self
   return o
end

--
function VBCMFA:new(...)
   local o = self:_factory()
   o:__init(...)
   return o
end

--
function VBCMFA:__init(cfg)
   local n, s, p, k, f, N = self:_setandgetDims(cfg)

   self.hidden = {
      Zm = torch.Tensor(s, k, n),
      Zcov = torch.Tensor(s, k, k):zeros(),

      Xm = torch.Tensor(s, f, n),
      Xcov = torch.Tensor(s, n, f, f):zeros(),

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

      E_starI = torch.Tensor(n, f, f),

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
      self.N = cfg.datasetSize
   end
   return self.n, self.s, self.p, self.k, self.f, self.N
end

--
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
function VBCMFA:_perpareForBatch(batchIdx)
   -- body
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

      local ELTPsiIL = torch.view(torch.view(Lcovt, p, k * k):t() * PsiI, k, k)
                           + LmtTPsiI * Lmt -- k x k

      torch.inverse(Zcovt, torch.eye(k) + ELTPsiIL) -- k x k
      Zm[t]:mm(Zcovt, LmtTPsiInY) -- k x k * k x n
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
      local Xcovt = Xcov[t] -- n x f x f
      local Qst = Qs[{ {}, t }] -- n
      local fQst = Qst:repeatTensor(f, 1) -- f x n
      local XmtfQst = torch.cmul(Xmt, fQst) -- f x n

      local nY = Y - Lm[t] * Zm[t] -- p x n
      local PsiInYXmtfQstT = torch.diag(PsiI) * nY * XmtfQst:t() -- p x f

      local EXcovtQs = torch.view(torch.view(Xcovt, n, f * f):t() * Qst, f, f) -- f x f
      local QstExxT = EXcovtQs + Xmt * XmtfQst:t() -- f x f
      local alpha_betat = torch.div(beta[t], alpha):cinv() -- f

      for q = 1, p do
         torch.inverse(Gcov[t][q], torch.diag(alpha_betat) + PsiI[q] * QstExxT) -- inv{ k x k + 1 * k x k }
         Gm[t][q]:mv(Gcov[t][q], PsiIYnXmtfQstT[q]) -- k x k * k x 1 = k x 1 
      end
   end
end

--
function VBCMFA:inferQX(X_star) -- f x n
   local n, s, p, k, f = self:_setandgetDims()
   local Gm, Gcov = self.factorLoading:getG()
   local Xm, Xcov = self.hidden:getX()
   local Lm = self.factorLoading:getL()
   local Zm = self.hidden:getZ()
   local E_starI = self.hyper.E_starI -- n x f x f

   for t = 1, s do
      local Gmt = Gm[t] -- p x k
      local Gcovt = Gcov[t] -- p x k x k
      local Xcovt = Xcov[t] -- n x f x f

      local GmtTPsiI = Gmt:t() * PsiI -- f x p
      

      local EGTPsiIG = torch.view(torch.view(Gcovt, p, f * f):t() * PsiI, f, f)
                           + GmtTPsiI * Gmt -- f x f


      local E_starIEGTPsiIG_split = (E_starI + EGTPsiIG:view(1, f, f):expand(n, f, f)):split(1) -- n x f x f, expand doesn't allocate new memory
      for i, XcovtiI in ipairs(E_starIEGTPsiIG_split) do
         Xcovt[i] = XcovtiI:inverse() -- f x f
      end
         
      local X_star3D = X_star:t():view(n, f, 1) -- n x f x 1
      local E_starIX_star3D = torch.baddbmm(E_starI, X_star3D)
      --                             n x [ f x f  * f x 1   ] = n x f x 1
  
      local nY = Y - Lm[t] * Zm[t] -- p x n
      local GmtTPsiInY = GmtTPsiI * nY -- f x n

      Xm[t] = torch.bmm(Xcovt, E_starIX_star3D + GmtTPsiInY:t()):squeeze():t() -- f x n
      --               n x [ f x f * f x 1 ] = n x f x 1
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
   local n, s, p, k, f = self:_setandgetDims()

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
function VBCMFA:inferPsiI(Y)
   local n, s, p, k, f = self:_setandgetDims()
   local Gm, Gcov = self.factorLoading:getG()
   local Xm, Xcov = self.hidden:getX()
   local Lm, Lcov = self.factorLoading:getL()
   local Zm, Zcov = self.hidden:getZ()
   local PsiI = self.hyper.PsiI

   local psi = torch.zeros(p, p)

   for t = 1, s do
      local Qst = Qs[{ {}, t }]
      local Zmt = Zm[t]
      local Xmt = Xm[t]
      local Xcovt = Xcov[t] -- n x f x f
      local Lmt = Lm[t]
      local Gmt = Gm[t]

      local kQst = Qst:repeatTensor(k, 1) -- k x n
      local EzzT = Zcov[t] * torch.sum(Qst) + Zmt * torch.cmul(Zmt, kQst):t() -- k x k
      
      local fQst = Qst:repeatTensor(f, 1) -- f x n
      
      local EXcovtQs = torch.view(torch.view(Xcovt, n, f * f):t() * Qst, f, f) -- f x f
      local ExxT = EXcovtQs + Xmt * torch.cmul(Xmt, fQst):t() -- f x f

      local EzxT = Zmt * Xmt:T() -- k x f
      local ELzxTGT = Lmt * EzxT * Gmt:t() -- p x p

      local pQst = Qst:repeatTensor(p, 1) -- p x n
      local YpQst = torch.cmul(pQst, Y) -- p x n

      local partialPsi = YpQst * (Y - 2 * Lmt * Zmt - 2 * Gmt * Xmt) -- 2 times because ultimately we 
                        + Lmt * EzzT * Lmt:t()                       -- are concerned with the diagonal
                        + Gmt * ExxT * Gmt:t()                       -- elements only
                        + 2 * ELzxTGT                                -- same reason here
      psi:add(partialPsi)
      for q = 1, p do
         psi[q][q] = psi[q][q] + torch.trace(Lcov[t][q] * EzzT) + torch.trace(Gcov[t][q] * ExxT)
      end
   end

   torch.div(PsiI, torch.diag(psi), n)
   PsiI:cinv()
end

--
function VBCMFA:inferab( ... )
   -- not implemented yet
end

--
function VBCMFA:inferalphabeta( ... )
   -- not implemented yet
end

--
function VBCMFA:inferPhi( ... )
   -- not implemented yet
end

--
function VBCMFA:inferE_starI(X_star) -- f x n
   local n, s, p, k, f = self:_setandgetDims()
   local Qs = self.hidden:getS()
   local Xm, Xcov = self.hidden:getX()

   local X_star3D = X_star:t():view(n, f, 1) -- n x f x 1
   local EX_starX_starT = torch.bmm(X_star3D, X_star3D:transform(2, 3)) -- n x f x f

   local Xcov_strt =  Xcov:transform(1, 2):view(n, s, f * f):transform(2, 3) -- n x (f*f) x s
   local EXXT = torch.bmm(Xcov_strt, Qs:view(n, s, 1)):view(n, f, f) -- n x f x f

   local EX = torch.bmm(Xm:transform(1, 3), Qs:view(n, s, 1)) -- n x f x 1

   local EX_starXT = torch.bmm(X_star3D, EX:transform(2, 3)) -- n x f x f

   local E_star = EXXT - EX_starXT - EX_starXT:transform(2, 3) + EX_starX_starT -- n x f x f

   for i, E_stari in ipairs(E_star:split(1)) do
      E_starI[i] = E_stari:inverse() -- f x f
   end
end