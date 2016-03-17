require 'cephes'

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
      Zm = torch.ones(s, k, n) / k,
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
      Xcov = torch.eye(f, f):repeatTensor(s, n, 1, 1),

      getX = function(self)
         return self.Xm, self.Xcov
      end
   }

   self.factorLoading = {
      Lm = torch.rand(s, p, k),
      Lcov = torch.eye(k, k):repeatTensor(s, p, 1, 1),

      a = 1,
      b = torch.rand(s, k),

      Gm = torch.rand(s, p, f),
      Gcov = torch.eye(f, f):repeatTensor(s, p, 1, 1),

      alpha = 1,
      beta = torch.rand(s, f),

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

      E_starI = torch.eye(f, f):repeatTensor(n, 1, 1),

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
   -- self._rho = 1

   print("finish init")
end

function VBCMFA:rho(t)
   if t then self.t = t
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

      if debug then
         print(string.format("nY for %d", t))
         print(nY)

         print(string.format("Gmt * Xmt for %d", t))
         print(Gm[t] * Xm[t])

         print(string.format("Gmt"))
         print(Gm[t])

         print(string.format("Xmt"))
         print(Zm[t])

         print("PsiInYZmtkQstT")
         print(PsiInYZmtkQstT)

         print("ZmtkQst")
         print(ZmtkQst)

         print("PsiInY")
         print(torch.diag(PsiI) * nY)
      end

      local QstEzzT = Zcov[t] * torch.sum(Qst) + Zmt * ZmtkQst:t() -- k x k
      local a_bt = torch.div(b[t], a):pow(-1) -- k

      if debug then
         print("QstEzzT")
         print(QstEzzT)

         print("a_bt")
         print(a_bt)
      end

      for q = 1, p do
         local rhoLcovtq = torch.inverse(torch.diag(a_bt) + QstEzzT * PsiI[q]) * rho -- k x k

         if debug then
            print("Lcovtq prev")
            print(Lcov[t][q])
         end

         Lcov[t][q]:add(rhoLcovtq, 1 - rho, Lcov[t][q])
         -- torch.inverse(Lcov[t][q], torch.diag(a_bt) + QstEzzT * PsiI[q]) -- inv{ k x k + 1 * k x k }

         if debug then
            print("Lcovtq result")
            print(Lcov[t][q])
            print("Lmtq prev")
            print(Lm[t][q])
         end

         Lm[t][q]:addmv(1 - rho, Lm[t][q], rho, Lcov[t][q], PsiInYZmtkQstT[q]) -- k x k * k x 1 = k x 1

         if debug then
            print("Lmtq result")
            print(Lm[t][q])
         end
      end
   end
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

      if debug then
         print("Lmt")
         print(Lmt)

         print("LmtTPsiI")
         print(LmtTPsiI)
      end

      local nY = Y - Gm[t] * Xm[t] -- p x n
      local LmtTPsiInY = LmtTPsiI * nY -- k x n

      if debug then
         print("nY")
         print(nY)

         print("Gmt * Xmt")
         print(Gm[t] * Xm[t])

         print("Gmt")
         print(Gm[t])

         print("Xmt")
         print(Xm[t])

         print("LmtTPsiInY")
         print(LmtTPsiInY)
      end

      local ELTPsiIL = torch.view(torch.view(Lcovt, p, k * k):t() * PsiI, k, k)
                           + LmtTPsiI * Lmt -- k x k

      if debug then
         print("Zcovt previous")
         print(Zcovt)
      end

      local rhoZcovt = torch.inverse(torch.eye(k) + ELTPsiIL) * rho
      Zcovt:add(rhoZcovt, 1 - rho, Zcovt)

      if debug then
         print("Zcovt result")
         print(Zcovt)

         print("Zmt previous")
         print(Zm[t])
      end

      -- Zm[t]:mm(Zcovt, LmtTPsiInY) -- k x k * k x n
      local rhoZmt = torch.mm(Zcovt, LmtTPsiInY) * rho
      Zm[t]:mul(1 - rho)
      Zm[t]:add(rhoZmt)

      if debug then
         print("Zmt Result")
         print(Zm[t])
      end
   end
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
      local Xcovt = Xcov[t] -- n x f x f
      local Qst = Qs[{ {}, t }] -- n

      local fQst = Qst:contiguous():view(1, n):expand(f, n) -- f x n
      local XmtfQst = torch.cmul(Xmt, fQst) -- f x n

      local nY = Y - Lm[t] * Zm[t] -- p x n
      local PsiInYXmtfQstT = torch.diag(PsiI) * nY * XmtfQst:t() -- p x f

      local EXcovtQs = torch.view(torch.view(Xcovt, n, f * f):t() * Qst, f, f) -- f x f
      local QstExxT = EXcovtQs + Xmt * XmtfQst:t() -- f x f
      local alpha_betat = torch.div(beta[t], alpha):pow(-1) -- f

      if debug then
         print(string.format("XmtfQst for %d", t))
         print(Xmt)

         print(string.format("Y for %d", t))
         print(Y)

         print(string.format("Lmt * Zmt for %d", t))
         print(Lm[t] * Zm[t])

         print(string.format("Lmt"))
         print(Lm[t])

         print(string.format("Zmt"))
         print(Zm[t])

         print(string.format("nY for %d", t))
         print(nY)

         print(string.format("PsiI for %d", t))
         print(torch.diag(PsiI))
      end

      -- Much faster version using approximation for inverse
      -- But this doesnt create positive definite Gcov matrix
      -- So be careful !!
      --
      -- local QstExxTI = torch.inverse(QstExxT) -- f x f
      -- local AIBAI = QstExxTI * torch.diag(alpha_betat) * QstExxTI -- f x f
      -- for q = 1, p do
      --    Gcov[t][q] = QstExxTI / PsiI[q] - AIBAI
      --    Gm[t][q]:mv(Gcov[t][q], PsiInYXmtfQstT[q]) -- f x f * f x 1 = f x 1
      -- end
      for q = 1, p do
         -- torch.inverse(Gcov[t][q], torch.diag(alpha_betat) + QstExxT * PsiI[q]) -- inv{ f x f + f x f * 1 }
         local rhoGcovtq = torch.inverse(torch.diag(alpha_betat) + QstExxT * PsiI[q]) * rho
         Gcov[t][q]:add(rhoGcovtq, 1 - rho, Gcov[t][q])
         Gm[t][q]:addmv(1 - rho, Gm[t][q], rho,  Gcov[t][q], PsiInYXmtfQstT[q]) -- f x f * f x 1 = f x 1
      end

      if debug then
         print(string.format("Gm for %d", t))
         print(Gm[t])
      end
   end
end

--
function VBCMFA:inferQX(debug, Y, X_star) -- f x n
   local n, s, p, k, f = self:_setandgetDims()
   local Gm, Gcov = self.factorLoading:getG()
   local Xm, Xcov = self.conditional:getX()
   local Lm = self.factorLoading:getL()
   local Zm = self.hidden:getZ()
   local E_starI = self.hyper.E_starI -- n x f x f
   local PsiI = self.hyper.PsiI
   local rho = self:rho()

   for t = 1, s do
      local Gmt = Gm[t] -- p x k
      local Gcovt = Gcov[t] -- p x k x k
      local Xcovt = Xcov[t] -- n x f x f

      local GmtTPsiI = Gmt:t() * torch.diag(PsiI) -- f x p

      local EGTPsiIG = torch.view(torch.view(Gcovt, p, f * f):t() * PsiI, f, f)
                           + GmtTPsiI * Gmt -- f x f

      -- if debug then
      --    print(string.format("Gmt for %d = ", t))
      --    print(Gmt)

      --    print("GmtTPsiI")
      --    print(GmtTPsiI)
      -- end

      local E_starIEGTPsiIG_split = (E_starI + EGTPsiIG:view(1, f, f):expand(n, f, f)):split(1) -- n x f x f, expand doesn't allocate new memory

      if debug then
         print("Previous Xcovt")
         print(Xcovt[1])
      end

      for i, XcovtiI in ipairs(E_starIEGTPsiIG_split) do
         local rhoXcovti = torch.inverse(XcovtiI:squeeze()) * rho
         Xcovt[i]:add(rhoXcovti, 1 - rho, Xcovt[i]) -- f x f
         -- print(string.format("Xcot[%d][%d] = ", t, i))
         -- print(XcovtiI)
         -- torch.inverse(Xcovt[i], XcovtiI:squeeze()) -- f x f
      end

      if debug then
         print("New Xcovt")
         print(Xcovt[1])
      end

      local X_star3D = X_star:view(f, n, 1):transpose(1, 2) -- n x f x 1
      local E_starIX_star3D = torch.bmm(E_starI, X_star3D)
      --                             n x [ f x f  * f x 1   ] = n x f x 1

      local nY = Y - Lm[t] * Zm[t] -- p x n
      local GmtTPsiInY = GmtTPsiI * nY -- f x n

      -- if debug then
      --    print(string.format("nY for %d", t))
      --    print(nY)

      --    print(string.format("Lmt * Zmt for %d", t))
      --    print(Lm[t] * Zm[t])

      --    print(string.format("Lmt"))
      --    print(Lm[t])

      --    print(string.format("Zmt"))
      --    print(Zm[t])

      --    print("GmtTPsiInY")
      --    print(GmtTPsiInY)

      --    print("E_starIX_star3D")
      --    print(E_starIX_star3D)

      --    print("Previous Xmt")
      --    print(Xm[t])
      -- end

      local Xmt_old = Xm[t]:view(f, n, 1):transpose(1, 2)
      Xm[t] = torch.baddbmm(1 - rho, Xmt_old, rho, Xcovt, E_starIX_star3D + GmtTPsiInY:t()):squeeze():t()
      --               n x [ f x f * f x 1 ] = n x f x 1
      -- Xm[t] = torch.bmm(Xcovt, E_starIX_star3D + GmtTPsiInY:t()):squeeze():t() -- f x n

      -- if debug then
      --    print("Result Xmt")
      --    print(Xm[t])
      -- end
   end
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
      beta[t] = beta[t] * (1 - rho) + (torch.ones(f) * beta_star + EL_sqr * 0.5) * rho
   end
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
      local Xmt = Xm[t] -- f x n
      local Zmt = Zm[t] -- k x n
      local Lmt = Lm[t] -- p x k
      local Gmt = Gm[t] -- p x f
      local Lcovt = Lcov[t] -- p x k x k
      local Gcovt = Gcov[t] -- p x f x f
      local Zcovt = Zcov[t] -- k x k
      local Xcovt = Xcov[t] -- n x f x f
      local PsiI_M = torch.diag(PsiI) -- p x p
      local logQst = logQs[{ {}, t }]

      local ELTPsiIG = Lmt:t() * PsiI_M * Gmt -- k x f
      local EzTLTPsiIGx = torch.sum(torch.cmul(Zmt, ELTPsiIG * Xmt), 1) -- 1 x n

      local ELTPsiIL = torch.view(torch.view(Lcovt, p, k * k):t() * PsiI, k, k)
                        + Lmt:t() * PsiI_M * Lmt -- k x k
      local EzTLTPsiILz = torch.sum(torch.cmul(Zmt, ELTPsiIL * Zmt), 1) -- 1 x n
                           + (torch.view(ELTPsiIL, 1, k * k) * torch.view(Zcovt:t():contiguous(), k * k, 1)):squeeze() -- 1

      local EGTPsiIG = torch.view(torch.view(Gcovt, p, f * f):t() * PsiI, f, f) -- f x f
                        + Gmt:t() * PsiI_M * Gmt -- f x f
      local ExTGTPsiIGx = torch.sum(torch.cmul(Xmt, EGTPsiIG * Xmt), 1) -- 1 x n
                           + torch.view(Xcovt, n, f * f) * torch.view(EGTPsiIG:t():contiguous(), f * f, 1) -- n x 1

      local nY = Y - Lmt * Zmt * 2 - Gmt * Xmt * 2 -- p x n
      local PsiInY = torch.diag(PsiI) * nY -- p x n

      for i = 1, n do
         logQst[i] = torch.sum(torch.log(torch.diag(torch.potrf(Xcovt[i], 'U'))))
      end

      logQst:add( - torch.sum(torch.cmul(Y, PsiInY), 1) * 0.5 -- 1 x n automatically converted to n x 1 while assignment
                  - EzTLTPsiIGx
                  - EzTLTPsiILz * 0.5
                  - ExTGTPsiIGx * 0.5
                  + torch.sum(torch.log(torch.diag(torch.potrf(Zcovt, 'U')))))
   end

   logQs:add(cephes.digamma(phim):float():view(1, s):expand(n, s))
   logQs = logQs - torch.max(logQs, 2) * torch.ones(1, s)
   torch.exp(Qs, logQs)
   Qs:cmul(torch.sum(Qs, 2):pow(-1) * torch.ones(1, s)) -- normalize

   -- if removal is enabled then
   -- then calculate the expected size (number of datapoints)
   -- of each component
   if calc then self._sizeofS:add(Qs:sum(1)) end
end

--
function VBCMFA:inferQpi()
   local n, s, p, k, f = self:_setandgetDims()
   local Qs, phim = self.hidden:getS()
   local rho = self:rho()

   local phi_starm = torch.ones(s) * self.hyper.phi_star / s
   local rhophim = torch.add(phi_starm, torch.sum(Qs, 1)[1]) * rho
   phim:mul(1 - rho)
   phim:add(rhophim)
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
      local Xcovt = Xcov[t] -- n x f x f
      local Lmt = Lm[t]
      local Gmt = Gm[t]

      local kQst = Qst:contiguous():view(1, n):expand(k, n) -- -- k x n
      local EzzT = Zcov[t] * torch.sum(Qst) + Zmt * torch.cmul(Zmt, kQst):t() -- k x k

      local fQst = Qst:contiguous():view(1, n):expand(f, n) -- -- f x n
      local EXcovtQs = torch.view(torch.view(Xcovt, n, f * f):t() * Qst, f, f) -- f x f
      local ExxT = EXcovtQs + Xmt * torch.cmul(Xmt, fQst):t() -- f x f

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

   local X_star3D = X_star:view(f, n, 1):transpose(1, 2) -- n x f x 1
   local EX_starX_starT = torch.bmm(X_star3D, X_star3D:transpose(2, 3)) -- n x f x f


   local Xcov_strt = Xcov:view(s, n, f * f):permute(2, 3, 1) -- n x (f*f) x s
   local EXXT = torch.bmm(Xcov_strt, Qs:view(n, s, 1)):view(n, f, f) -- n x f x f

   local EX = torch.bmm(Xm:transpose(1, 3), Qs:view(n, s, 1)) -- n x f x 1

   local EX_starXT = torch.bmm(X_star3D, EX:transpose(2, 3)) -- n x f x f

   local E_star = EXXT - EX_starXT - EX_starXT:transpose(2, 3) + EX_starX_starT -- n x f x f
      if debug then
         print(string.format("Old E_starI[%d]", 1))
         print(E_starI[1])
      end

   for i, E_stari in ipairs(E_star:split(1)) do

      E_starI[i]:mul(1 - rho)
      E_starI[i]:add(torch.inverse(E_stari:squeeze()) * rho) -- f x f

   end
      if debug then
         print(string.format("New E_starI[%d]", 1))
         print(E_starI[1])
      end
end

--
function VBCMFA:calcF()
   -- body
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

return VBCMFA