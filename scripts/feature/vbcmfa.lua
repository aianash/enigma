local pl = (require 'pl.import_into')()
require 'cephes'

function kldirichlet(phimP, phimQ)
   phimP0 = torch.sum(phimP)
   phimQ0 = torch.sum(phimQ)

   return (cephes.lgam(phimP0) - cephes.lgam(phimQ0)
            - torch.sum(cephes.lgam(phimP) - cephes.lgam(phimQ))
            + (phimP - phimQ):double():dot(cephes.digamma(phimP) - cephes.digamma(phimQ)))
end

function klgamma(pa, pb, qa, qb)
   return torch.sum(
              (pa - qa):double():cmul(cephes.digamma(pa))
            - cephes.lgam(pa) + cephes.lgam(qa)
            + (torch.log(pb) - torch.log(qb)):cmul(qa):double()
            + (pb - qb):cmul(pa):cdiv(pb):double())
end

-- Calculate the nearest positive definite matrix
function posdefify(M, ev)
   local e, V = torch.symeig(M, 'V')
   local n = e:size(1)
   local ev = ev or 1e-7
   local eps = ev * math.abs(e[n])

   if e[1] < eps then
      e[e:lt(eps)] = eps
      local old = torch.diag(M)
      local Mnew = V * torch.diag(e) * V:t()
      local D = old:cmax(eps):cdiv(torch.diag(Mnew)):sqrt()
      M:copy(torch.diag(D) * Mnew * torch.diag(D))
   end
end

function logdet(M)
   local status, logdet = pcall(function()
      return 2 * torch.sum(torch.log(torch.diag(torch.potrf(M, 'U'))))
   end)

   if not status then
      posdefify(M)
      logdet = 2 * torch.sum(torch.log(torch.diag(torch.potrf(M, 'U'))))
   end
   return logdet
end

--
function inverse(M, definite, ev)
   local definite = definite or true
   local status, inv = pcall(function ()
      return torch.inverse(M)
   end)
   local ev = ev or 1e-7

   if not status then
      local u, s, v = torch.svd(M)
      local tol = ev * M:size(1) * s[1]
      s[s:lt(tol)] = 0
      s:pow(-1)
      s[s:eq(math.huge)] = 0
      inv = u * torch.diag(s) * v:t()
      print(string.format("INV = %s", inv))
   end

   if definite then posdefify(inv) end
   return inv
end

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
-- E_starI    - f x f
--
-- phi_star   - 1                        (a number)
--
-- PsiI       - p                        (a diagonal matrix)
----------------------
local VBCMFA = {}

--
function VBCMFA:_factory()
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
      Zm = torch.randn(s, k, n),
      Zcov = torch.eye(k, k):repeatTensor(s, 1, 1),

      Qs = torch.ones(n, s) / s,
      phim = torch.ones(s) / s,

      getZ = function(self)
         return self.Zm, self.Zcov
      end,

      getS = function(self)
         return self.Qs, self.phim
      end
   }

   self.conditional = {
      Xm = torch.randn(s, f, n),
      Xcov = torch.eye(f, f):repeatTensor(s, 1, 1),

      getX = function(self)
         return self.Xm, self.Xcov
      end
   }

   self.factorLoading = {
      Lm = torch.randn(s, p, k),
      Lcov = torch.eye(k, k):repeatTensor(s, p, 1, 1),

      a = 1,
      b = torch.randn(s, k),

      Gm = torch.randn(s, p, f),
      Gcov = torch.eye(f, f):repeatTensor(s, p, 1, 1),

      alpha = 1,
      beta = torch.randn(s, f),

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

      E_starI = torch.eye(f, f),

      phi_star = 1,

      PsiI = torch.ones(p),

      get = function(self)
         return self.a_star, self.b_star, self.alpha_star, self.beta_star, self.E_starI, self.phi_star, self.PsiI
      end
   }

   self.lr = cfg.learningRate

   --
   self._sizeofS = torch.randn(s)
   self.removal = cfg.removal

   self.tau = cfg.delay
   self.kappa = cfg.forgettingRate
   self.Fmatrix = torch.zeros(9, s)
   self.F = - 1 / 0
   self._rho = 0.5
   print("finish init")
end

function VBCMFA:reset()
   local n, s, p, k, f = self:_setandgetDims()

   print("Resetting parameters")
   self.hidden.Zm = torch.randn(s, k, n)
   self.hidden.Zcov = torch.eye(k, k):repeatTensor(s, 1, 1)

   self.hidden.Qs = torch.ones(n, s) / s
   self.hidden.phim = torch.ones(s) / s

   self.conditional.Xm = torch.randn(s, f, n)
   self.conditional.Xcov = torch.eye(f, f):repeatTensor(s, 1, 1)

   self.factorLoading.Lm = torch.randn(s, p, k)
   self.factorLoading.Lcov = torch.eye(k, k):repeatTensor(s, p, 1, 1)

   self.factorLoading.a = 1
   self.factorLoading.b = torch.randn(s, k)

   self.factorLoading.Gm = torch.randn(s, p, f)
   self.factorLoading.Gcov = torch.eye(f, f):repeatTensor(s, p, 1, 1)

   self.factorLoading.alpha = 1
   self.factorLoading.beta = torch.randn(s, f)

   self.hyper.a_star = 1
   self.hyper.b_star = 1
   self.hyper.alpha_star = 1
   self.hyper.beta_star = 1
   self.hyper.E_starI = torch.eye(f, f)
   self.hyper.phi_star = 1
   self.hyper.PsiI = torch.ones(p)

   self._sizeofS = torch.zeros(s)
end

--
function VBCMFA:rho(t)
   if t == 1 then self._rho = 1
   elseif t then self._rho = 0.5; self.t = t
   elseif self._rho then return self._rho
   else return (self.t + self.tau) ^ -self.kappa end
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
function VBCMFA:_checkDimensions(tensor, ...)
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

------------
--[[ Lz ]]--
------------

--
function VBCMFA:inferQL(debug, Y) -- p x n
   local n, s, p, k, f = self:_setandgetDims()
   local Lm, Lcov, a, b = self.factorLoading:getL()
   local Gm, Gcov = self.factorLoading:getG()
   local Zm, Zcov = self.hidden:getZ()
   local Xm = self.conditional:getX()
   local Qs = self.hidden:getS()
   local PsiI = self.hyper.PsiI
   local rho = self:rho()

   for t = 1, s do
      local Zmt = Zm[t] -- k x n
      local Qst = Qs[{ {}, t }]
      local kQst = Qst:contiguous():view(1, n):expand(k, n) -- -- k x n
      local ZmtkQst = torch.cmul(Zmt, kQst) -- k x n

      local nY = Y - Gm[t] * Xm[t] -- p x n

      local PsiInYZmtkQstT = torch.diag(PsiI) * nY * ZmtkQst:t() -- p x k

      local QstEzzT = Zcov[t] * torch.sum(Qst) + Zmt * ZmtkQst:t() -- k x k
      local a_bt = torch.div(b[t], a):pow(-1) -- k

      for q = 1, p do
         local rhoLcovtq = inverse(torch.diag(a_bt) + QstEzzT * PsiI[q]) * rho -- k x k
         Lcov[t][q]:add(rhoLcovtq, 1 - rho, Lcov[t][q])
         Lm[t][q]:addmv(1 - rho, Lm[t][q], rho, Lcov[t][q], PsiInYZmtkQstT[q]) -- k x k * k x 1 = k x 1
      end
   end

   self:check(Lm, "Lm")
   self:check(Lcov, "Lcov")
end

--
function VBCMFA:inferQZ(debug, Y)
   local n, s, p, k, f = self:_setandgetDims()
   local Lm, Lcov, a, b = self.factorLoading:getL()
   local Gm = self.factorLoading:getG()
   local Zm, Zcov = self.hidden:getZ()
   local Xm = self.conditional:getX()
   local PsiI = self.hyper.PsiI
   local rho = self:rho()

   for t = 1, s do
      local Lmt = Lm[t]
      local Lcovt = Lcov[t]
      local Zcovt = Zcov[t]

      local LmtTPsiI = Lmt:t() * torch.diag(PsiI) -- k x p

      local nY = Y - Gm[t] * Xm[t] -- p x n
      local LmtTPsiInY = LmtTPsiI * nY -- k x n

      local ELTPsiIL = torch.view(torch.view(Lcovt, p, k * k):t() * PsiI, k, k)
                           + LmtTPsiI * Lmt -- k x k

      local rhoZcovt = inverse(torch.eye(k) + ELTPsiIL) * rho
      Zcovt:add(rhoZcovt, 1 - rho, Zcovt)

      local rhoZmt = torch.mm(Zcovt, LmtTPsiInY) * rho
      Zm[t]:mul(1 - rho)
      Zm[t]:add(rhoZmt)
   end

   self:check(Zm, "Zm")
   self:check(Zcov, "Zcov")
end

--
function VBCMFA:inferQnu()
   local n, s, p, k, f = self:_setandgetDims()
   local Lm, Lcov, a, b = self.factorLoading:getL()
   local a_star, b_star = self.hyper.a_star, self.hyper.b_star
   local rho = self:rho()

   self.factorLoading.a = (1 - rho) * a + rho * (a_star + p / 2)
   for t = 1, s do
      local Lmt = Lm[t] -- p x k
      local Lcovt = Lcov[t] -- p x k x k
      local EL_sqr = torch.diag(torch.sum(Lcovt, 1)[1]) + torch.sum(torch.pow(Lmt, 2), 1)[1] -- k
      b[t] = b[t] * (1 - rho) + (torch.ones(k) * b_star + EL_sqr * 0.5) * rho
   end

   self:check(b, "b")
end

------------
--[[ Gx ]]--
------------

--
function VBCMFA:inferQG(debug, Y)
   local n, s, p, k, f = self:_setandgetDims()
   local Gm, Gcov, alpha, beta = self.factorLoading:getG()
   local Lm, Lcov = self.factorLoading:getL()
   local Xm, Xcov = self.conditional:getX()
   local Zm = self.hidden:getZ()
   local Qs = self.hidden:getS()
   local PsiI = self.hyper.PsiI
   local rho = self:rho()

   for t = 1, s do
      local Xmt = Xm[t] -- f x n
      local Qst = Qs[{ {}, t }] -- n

      local fQst = Qst:contiguous():view(1, n):expand(f, n) -- f x n
      local XmtfQst = torch.cmul(Xmt, fQst) -- f x n

      local nY = Y - Lm[t] * Zm[t] -- p x n
      local PsiInYXmtfQstT = torch.diag(PsiI) * nY * XmtfQst:t() -- p x f

      local QstExxT = Xcov[t] * torch.sum(Qst) + Xmt * XmtfQst:t() -- f x f
      local alpha_betat = torch.div(beta[t], alpha):pow(-1) -- f

      -- Much faster version using approximation for inverse
      -- But this doesnt create positive definite Gcov matrix
      -- So be careful !!
      --
      -- local QstExxTI = inverse(QstExxT) -- f x f
      -- local AIBAI = QstExxTI * torch.diag(alpha_betat) * QstExxTI -- f x f
      -- for q = 1, p do
      --    Gcov[t][q] = QstExxTI / PsiI[q] - AIBAI
      --    Gm[t][q]:mv(Gcov[t][q], PsiInYXmtfQstT[q]) -- f x f * f x 1 = f x 1
      -- end
      for q = 1, p do
         local rhoGcovtq = inverse(torch.diag(alpha_betat) + QstExxT * PsiI[q]) * rho
         Gcov[t][q]:add(rhoGcovtq, 1 - rho, Gcov[t][q])
         Gm[t][q]:addmv(1 - rho, Gm[t][q], rho,  Gcov[t][q], PsiInYXmtfQstT[q]) -- f x f * f x 1 = f x 1
      end
   end

   self:check(Gm, "Gm")
   self:check(Gcov, "Gcov")
end

--
function VBCMFA:inferQX(debug, Y, X_star) -- f x n
   local n, s, p, k, f = self:_setandgetDims()
   local Gm, Gcov = self.factorLoading:getG()
   local Xm, Xcov = self.conditional:getX()
   local Lm = self.factorLoading:getL()
   local Zm = self.hidden:getZ()
   local E_starI = self.hyper.E_starI -- f x f
   local PsiI = self.hyper.PsiI
   local rho = self:rho()

   for t = 1, s do
      local Gmt = Gm[t] -- p x k
      local Gcovt = Gcov[t] -- p x k x k
      local Xcovt = Xcov[t] -- n x f x f

      local GmtTPsiI = Gmt:t() * torch.diag(PsiI) -- f x p

      local EGTPsiIG = torch.view(torch.view(Gcovt, p, f * f):t() * PsiI, f, f)
                           + GmtTPsiI * Gmt -- f x f

      local rhoXcovt = inverse(E_starI + EGTPsiIG) * rho -- f x f
      Xcovt:add(rhoXcovt, 1 - rho, Xcov[t]) -- f x f

      local X_star3D = X_star:view(f, n, 1):transpose(1, 2) -- n x f x 1
      local E_starIX_star3D = torch.bmm(E_starI:view(1, f, f):expand(n, f, f), X_star3D)
      --                             n x [ f x f                             * f x 1   ] = n x f x 1

      local nY = Y - Lm[t] * Zm[t] -- p x n
      local GmtTPsiInY = GmtTPsiI * nY -- f x n

      local Xmt_old = Xm[t]:view(f, n, 1):transpose(1, 2)
      Xm[t] = torch.baddbmm(  1 - rho, Xmt_old,
                              rho, Xcovt:view(1, f, f):expand(n, f, f),
                              E_starIX_star3D + GmtTPsiInY:t()
                           ):view(n, f):t()
      --               n x [ f x f * f x 1 ] = n x f x 1
   end

   self:check(Xm, "Xm")
   self:check(Xcov, "Xcov")
end

--
function VBCMFA:inferQomega()
   local n, s, p, k, f = self:_setandgetDims()
   local Gm, Gcov, alpha, beta = self.factorLoading:getG()
   local alpha_star, beta_star = self.hyper.alpha_star, self.hyper.beta_star
   local rho = self:rho()

   self.factorLoading.alpha = (1 - rho) * alpha + rho * (alpha_star + p / 2)
   for t = 1, s do
      local Gmt = Gm[t] -- p x f
      local Gcovt = Gcov[t] -- p x f x f
      local EL_sqr = torch.diag(torch.sum(Gcovt, 1)[1]) + torch.sum(torch.pow(Gmt, 2), 1)[1] -- f
      self:check(beta[t], "oldbeta[t]")
      self:check((torch.ones(f) * beta_star + EL_sqr * 0.5), "EL_sqr")
      -- print(beta[t])
      -- print(EL_sqr)
      self:check(beta[t] * (1 - rho), "EL_sqr")
      beta[t] = beta[t] * (1 - rho) + (torch.ones(f) * beta_star + EL_sqr * 0.5) * rho
      self:check(beta[t], "afterbeta[t]")
   end

   self:check(beta, "beta")
end

-----------
--[[ S ]]--
-----------

--
function VBCMFA:inferQs(debug, Y, calc)
   local n, s, p, k, f = self:_setandgetDims()
   local Gm, Gcov = self.factorLoading:getG()
   local Xm, Xcov = self.conditional:getX()
   local Lm, Lcov = self.factorLoading:getL()
   local Zm, Zcov = self.hidden:getZ()
   local Qs, phim = self.hidden:getS()
   local PsiI = self.hyper.PsiI

   local logQs = torch.zeros(n, s) -- probably no need for extra memory here

   for t = 1, s do
      local Zmt = Zm[t] -- k x n
      local Xmt = Xm[t] -- f x n
      local Lmt = Lm[t] -- p x k
      local Gmt = Gm[t] -- p x f
      local Zcovt = Zcov[t] -- k x k
      local Xcovt = Xcov[t] -- f x f
      local Lcovt = Lcov[t] -- p x k x k
      local Gcovt = Gcov[t] -- p x f x f
      local PsiI_M = torch.diag(PsiI) -- p x p

      local ELTPsiIG = Lmt:t() * PsiI_M * Gmt -- k x f
      local EzTLTPsiIGx = torch.sum(torch.cmul(Zmt, ELTPsiIG * Xmt), 1) -- 1 x n

      local ELTPsiIL = torch.view(torch.view(Lcovt, p, k * k):t() * PsiI, k, k)
                        + Lmt:t() * PsiI_M * Lmt -- k x k
      local EzTLTPsiILz = torch.sum(torch.cmul(Zmt, ELTPsiIL * Zmt), 1) -- 1 x n
                           + (torch.view(ELTPsiIL, 1, k * k) * torch.view(Zcovt:t():contiguous(), k * k, 1)):squeeze() -- 1

      local EGTPsiIG = torch.view(torch.view(Gcovt, p, f * f):t() * PsiI, f, f) -- f x f
                        + Gmt:t() * PsiI_M * Gmt -- f x f
      local ExTGTPsiIGx = torch.sum(torch.cmul(Xmt, EGTPsiIG * Xmt), 1) -- 1 x n
                           + (torch.view(EGTPsiIG, 1, f * f) * torch.view(Xcovt:t():contiguous(), f * f, 1)):squeeze() -- 1

      local nY = Y - Lmt * Zmt * 2 - Gmt * Xmt * 2 -- p x n
      local PsiInY = torch.diag(PsiI) * nY -- p x n

      logQs[{ {}, t}] = - torch.sum(torch.cmul(Y, PsiInY), 1) * 0.5 -- 1 x n automatically converted to n x 1 while assignment
                  - EzTLTPsiIGx * 2 * 0.5
                  - EzTLTPsiILz * 0.5
                  - ExTGTPsiIGx * 0.5
                  + logdet(Zcovt) * 0.5
                  + logdet(Xcovt) * 0.5
   end

   logQs:add(cephes.digamma(phim):float():view(1, s):expand(n, s))
   logQs = logQs - torch.max(logQs, 2) * torch.ones(1, s)
   torch.exp(Qs, logQs)
   Qs:cmul(torch.sum(Qs, 2):pow(-1) * torch.ones(1, s)) -- normalize

   -- if removal is enabled thens
   -- then calculate the expected size (number of datapoints)
   -- of each component
   if calc then
      self._sizeofS:add(Qs:sum(1))
      print(string.format("Qs responsibility = %s", Qs:sum(1)))
   end

   self:check(Qs, "Qs")
end

--
function VBCMFA:inferQpi()
   local n, s, p, k, f = self:_setandgetDims()
   local Qs, phim = self.hidden:getS()
   local rho = self:rho()

   local phi_starm = torch.ones(s) * self.hyper.phi_star / s
   local rhophim = torch.add(phi_starm, torch.sum(Qs, 1):squeeze()) * rho
   phim:mul(1 - rho)
   phim:add(rhophim)
   self:check(phim, "phim")

   -- print(string.format("phim = %s", phim))
end

--------------------------
--[[ Hyper parameters ]]--
--------------------------

--
function VBCMFA:inferPsiI(debug, Y)
   local n, s, p, k, f = self:_setandgetDims()
   local Gm, Gcov = self.factorLoading:getG()
   local Xm, Xcov = self.conditional:getX()
   local Lm, Lcov = self.factorLoading:getL()
   local Qs = self.hidden:getS()
   local Zm, Zcov = self.hidden:getZ()
   local PsiI = self.hyper.PsiI
   local rho = self:rho()

   local psi = torch.zeros(p, p)
   for t = 1, s do
      local Qst = Qs[{ {}, t }]
      local Zmt = Zm[t]
      local Xmt = Xm[t]
      local Lmt = Lm[t]
      local Gmt = Gm[t]

      local kQst = Qst:contiguous():view(1, n):expand(k, n) -- -- k x n
      local EzzT = Zcov[t] * torch.sum(Qst) + Zmt * torch.cmul(Zmt, kQst):t() -- k x k

      local fQst = Qst:contiguous():view(1, n):expand(f, n) -- -- f x n
      local ExxT = Xcov[t] * torch.sum(Qst) + Xmt * torch.cmul(Xmt, fQst):t() -- f x f

      local EzxT = Zmt * torch.cmul(Xmt, fQst):t() -- k x f
      local ELzxTGT = Lmt * EzxT * Gmt:t() -- p x p

      local pQst = Qst:contiguous():view(1, n):expand(p, n) -- -- p x n
      local YpQst = torch.cmul(pQst, Y) -- p x n

      local partialPsi = YpQst * (Y - Lmt * Zmt * 2 - Gmt * Xmt * 2):t() -- 2 times because ultimately we
                        + Lmt * EzzT * Lmt:t()                       -- are concerned with the diagonal
                        + Gmt * ExxT * Gmt:t()                       -- elements only
                        + ELzxTGT * 2                               -- same reason here
      psi:add(partialPsi)
      for q = 1, p do
         psi[q][q] = psi[q][q] + torch.trace(Lcov[t][q] * EzzT) + torch.trace(Gcov[t][q] * ExxT)
      end
   end

   PsiI:mul(1 - rho)
   PsiI:add(torch.div(torch.diag(psi), n):pow(-1) * rho)

   self:check(PsiI, "PsiI")
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
function VBCMFA:inferE_starI(debug, X_star) -- f x n
   local n, s, p, k, f = self:_setandgetDims()
   local Qs = self.hidden:getS()
   local Xm, Xcov = self.conditional:getX()
   local E_starI = self.hyper.E_starI
   local rho = self:rho()

   local Ps = torch.sum(Qs, 1)[1] -- s
   local E_star = torch.zeros(f, f)
   local X_star_Xm = X_star:view(1, f, n):expand(s, f, n) - Xm -- s x f x n

   for t = 1, s do
      local Xmt = Xm[t]
      local Qst = Qs[{ {}, t }]
      local X_star_Xmt = X_star_Xm[t]

      local fQst = Qst:contiguous():view(1, n):expand(f, n) -- f x n

      local X_star_XmtfQst = torch.cmul(X_star_Xmt, fQst) -- f x n
      E_star = E_star + Xcov[t] * Ps[t] + X_star_Xmt * X_star_XmtfQst:t() -- f x f
   end
   E_star:mul(1 / n)

   E_starI:mul(1 - rho)
   E_starI:add(inverse(E_star) * rho)
end

--
function VBCMFA:calcF(debug, Y, X_star) -- p x n, f x n
   local n, s, p, k, f = self:_setandgetDims()
   local Lm, Lcov, a, b = self.factorLoading:getL()
   local Gm, Gcov, alpha, beta = self.factorLoading:getG()
   local Zm, Zcov = self.hidden:getZ()
   local Xm, Xcov = self.conditional:getX()
   local Qs, phim = self.hidden:getS()
   local a_star, b_star, alpha_star, beta_star, E_starI, phi_star, PsiI = self.hyper:get()

   local Fmatrix = self.Fmatrix -- 7 x s
   local PsiI_M = torch.diag(self.hyper.PsiI)

   local Ps = torch.sum(Qs, 1)[1] -- s

   local logDetE_star = - logdet(E_starI)

   local X_star_Xm = X_star:view(1, f, n):expand(s, f, n) - Xm -- s x f x n
   local Qsmod = Qs:clone()
   Qsmod[Qs:eq(0)] = 1

   local digamphim = cephes.digamma(phim) -- s
   local digsumphim = cephes.digamma(torch.sum(phim)) -- 1

   Fmatrix1 = - kldirichlet(phim, torch.ones(s) * phi_star / s)

   local F_old = self.F

   --
   for t = 1, s do
      local Xmt = Xm[t]
      local Xcovt = Xcov[t]
      local Zmt = Zm[t]
      local Zcovt = Zcov[t]
      local Gmt = Gm[t]
      local Gcovt = Gcov[t]
      local Lmt = Lm[t]
      local Lcovt = Lcov[t]
      local Qst = Qs[{ {}, t }] -- n x 1
      local Qsmodt = Qsmod[{ {}, t }] -- n x 1
      local Fmatrixt = Fmatrix[{ {}, t }]
      local X_star_Xmt = X_star_Xm[t] -- f x n

      local logDet2piPsiI = - torch.log(PsiI):sum() + p * math.log(2 * math.pi) -- 1

      -- Fmatrix[2]
      Fmatrixt[2] = - klgamma(torch.ones(k) * a, b[t], torch.ones(k) * a_star, torch.ones(k) * b_star)

      -- Fmatrx[3]
      local a_bt = torch.div(b[t], a):pow(-1) -- k
      local f3 = cephes.digamma(a) * k - torch.sum(torch.log(b[t]))

      for q = 1, p do
         f3 = f3 - k
               + logdet(Lcovt[q])
               - (torch.diag(Lcovt[q]) + torch.pow(Lmt[q], 2)):dot(a_bt)
      end
      Fmatrixt[3] = f3 / 2

      -- Fmatrix[4]
      Fmatrixt[4] = - klgamma(torch.ones(f) * alpha, beta[t], torch.ones(f) * alpha_star, torch.ones(f) * beta_star)

      -- Fmatrix[5]
      local alpha_betat = torch.div(beta[t], alpha):pow(-1) -- k
      local f5 = cephes.digamma(alpha) * f - torch.sum(torch.log(beta[t]))

      for q = 1, p do
         f5 = f5 - f
               + logdet(Gcovt[q])
               - (torch.diag(Gcovt[q]) + torch.pow(Gmt[q], 2)):dot(alpha_betat)
      end
      Fmatrixt[5] = f5 / 2

      -- Fmatrix[6]
      Fmatrixt[6] = torch.sum(torch.cmul(Qst, - torch.log(Qsmodt) + torch.ones(n) * (digamphim[t] - digsumphim)))

      -- Fmatrix[7]
      local kQst = Qst:contiguous():view(1, n):expand(k, n) -- k x n
      local ZmtkQst = torch.cmul(Zmt, kQst) -- k x n
      local QstEzzT = Zcovt * Ps[t] + Zmt * ZmtkQst:t()
      Fmatrixt[7] = 0.5 * k * torch.sum(Qst)
                    + 0.5 * Ps[t] * logdet(Zcovt)
                    - 0.5 * torch.trace(QstEzzT)

      -- Fmatrix[8]
      local fQst = Qst:contiguous():view(1, n):expand(f, n) -- f x n
      local X_star_XmtfQst = torch.cmul(X_star_Xmt, fQst) -- f x n
      local QstExxT = Xcovt * Ps[t] + X_star_Xmt * X_star_XmtfQst:t() -- f x f

      Fmatrixt[8] = 0.5 * f * torch.sum(Qst)
                    - 0.5 * Ps[t] * logDetE_star
                    + 0.5 * Ps[t] * logdet(Xcovt)
                    - 0.5 * torch.trace(E_starI * QstExxT)

      -- Fmatrix[9]
      local ELTPsiIG = Lmt:t() * PsiI_M * Gmt -- k x f
      local EzTLTPsiIGx = torch.sum(torch.cmul(Zmt, ELTPsiIG * Xmt), 1) -- 1 x n

      local ELTPsiIL = torch.view(torch.view(Lcovt, p, k * k):t() * PsiI, k, k)
                           + Lmt:t() * PsiI_M * Lmt -- k x k
      local EzTLTPsiILz = torch.sum(torch.cmul(Zmt, ELTPsiIL * Zmt), 1) -- 1 x n
                           + (torch.view(ELTPsiIL, 1, k * k) * torch.view(Zcovt:t():contiguous(), k * k, 1)):squeeze() -- 1

      local EGTPsiIG = torch.view(torch.view(Gcovt, p, f * f):t() * PsiI, f, f) -- f x f
                        + Gmt:t() * PsiI_M * Gmt -- f x f
      local ExTGTPsiIGx = torch.sum(torch.cmul(Xmt, EGTPsiIG * Xmt), 1) -- 1 x n
                           + (torch.view(EGTPsiIG, 1, f * f) * torch.view(Xcovt:t():contiguous(), f * f, 1)):squeeze() -- 1

      local nY = Y - Lmt * Zmt * 2 - Gmt * Xmt * 2 -- p x n
      local PsiInY = torch.diag(PsiI) * nY -- p x n

      local E = - torch.sum(torch.cmul(Y, PsiInY), 1) * 0.5
                - EzTLTPsiIGx * 2 * 0.5
                - EzTLTPsiILz * 0.5
                - ExTGTPsiIGx * 0.5

      Fmatrixt[9] = torch.cmul(E, Qst):sum() - 0.5 * Ps[t] * logDet2piPsiI
   end

   self.F = torch.sum(Fmatrix) + Fmatrix1
   self.dF = self.F - F_old
   print(Fmatrix)
   return self.F, self.dF
end

--
function VBCMFA:doremoval()
   local sizeOfS = self._sizeofS

   if self.removal then
      local dying = sizeOfS[sizeofS:lt(1)]

      if dying:size(1) > 0 then


      end

   end
end

--
function VBCMFA:dobirth()
   -- body
end

--
function VBCMFA:check(X, name)
   local nDim = X:nDimension()
   local sizes = torch.Tensor(torch.Storage(nDim):copy(X:size()))
   local X1 = X:view(sizes:prod(), 1)

   local v, indices = torch.max(X1:ne(X1), 1)
   v = v:squeeze()

   if v == 1 then
      for i = 1, indices:size(1) do
         local lidx = indices[i]:squeeze()
         local idx = {}
         for d = 1, nDim - 1 do
            local sz = sizes:narrow(1, d + 1, nDim - d):prod()
            idx[d] = math.floor(lidx / sz + 1)
            lidx = lidx - (idx[d] - 1) * sz
         end
         idx[nDim] = math.floor(lidx)
         local str = name
         local x = X
         for _, i in ipairs(idx) do
            str = str.."["..tostring(i).."]"
            x = x[i]
         end
         print(string.format(str.." = %f", x))
      end
      os.exit()
   end
end

return VBCMFA