require 'torch'
require 'cephes'
require 'distributions'

torch.setdefaulttensortype('torch.FloatTensor')

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


function VBCMFA:__init(cfg)
   local n, s, p, k, d, N = self:_setandgetDims(cfg)

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
      Xm = torch.randn(s, d, n),
      Xcov = torch.eye(d, d):repeatTensor(s, 1, 1),

      getX = function(self)
         return self.Xm, self.Xcov
      end
   }

   self.factorLoading = {
      Lm = torch.randn(s, p, k),
      Lcov = torch.eye(k, k):repeatTensor(s, p, 1, 1),

      a = 1,
      b = torch.ones(s, k),

      Gm = torch.randn(s, p, d),
      Gcov = torch.eye(d, d):repeatTensor(s, p, 1, 1),

      alpha = 1,
      beta = torch.ones(s, d),

      getL = function(self)
         return self.Lm, self.Lcov, self.a, self.b
      end,

      getG = function(self)
         return self.Gm, self.Gcov, self.alpha, self.beta
      end
   }

   self.hyper = {
      mu_star = torch.randn(d, n),
      sigma_star = torch.eye(d, d),

      a_star = 1,
      b_star = 1,

      alpha_star = 1,
      beta_star = 1,

      phi_star = 1,

      PsiI = torch.ones(p),

      get = function(self)
         return self.a_star, self.b_star, self.alpha_star, self.beta_star, self.mu_star, self.sigma_star, self.phi_star, self.PsiI
      end
   }

   self.Fmatrix = torch.zeros(9, s)
   self.F = - 1 / 0
   self.hardness = cfg.hardness
end


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


---------------------------------------------
-- Pt : n x sn
-- Mu  : sn x p x n
---------------------------------------------
function VBCMFA:inferQs(Pt, Mu)
   local n, S, p, k, f = self:_setandgetDims()
   local Gm, Gcov = self.factorLoading:getG()
   local Xm, Xcov = self.conditional:getX()
   local Lm, Lcov = self.factorLoading:getL()
   local Zm, Zcov = self.hidden:getZ()
   local Qs, phim = self.hidden:getS()
   local PsiI = self.hyper.PsiI

   local sn = Pt:size(2)
   local pPt = Pt:view(n, sn, 1):expand(n, sn, p):permute(2, 3, 1)  -- sn x p x n

   local logQs = torch.zeros(n, S) -- n x S

   for s = 1, S do
      local Xms = Xm[s]  -- f x n
      local Zms = Zm[s]  -- k x n
      local Lms = Lm[s]  -- p x k
      local Gms = Gm[s]  -- p x f
      local Xcovs = Xcov[s]  -- n x f x f
      local Zcovs = Zcov[s]  -- k x k
      local Lcovs = Lcov[s]  -- p x k x k
      local Gcovs = Gcov[s]  -- p x f x f
      local PsiI_M = torch.diag(PsiI) -- p x p

      local ELTPsiIG = Lms:t() * PsiI_M * Gms  -- k x d
      local EzTLTPsiIGx = torch.sum(torch.cmul(Zms, ELTPsiIG * Xms), 1)  -- 1 x n

      local ELTPsiIL = (Lcovs:view(p, k * k):t() * PsiI):view(k, k)
                     + Lms:t() * PsiI_M * Lms
      local EzTLTPsiILz = torch.sum(torch.cmul(Zms, ELTPsiIL * Zms), 1) -- 1 x n
                        + (torch.view(ELTPsiIL, 1, k * k) * torch.view(Zcovs:t():contiguous(), k * k, 1)):squeeze() -- 1

      local EGTPsiIG = (Gcovs:view(p, f * f):t() * PsiI):view(f, f) -- f x f
                     + Gms:t() * PsiI_M * Gms
      local ExTGTPsiIGx = torch.sum(torch.cmul(Xms, EGTPsiIG * Xms), 1) -- 1 x n
                        + (torch.view(EGTPsiIG, 1, f * f) * torch.view(Xcovs:t():contiguous(), f * f, 1)):squeeze() -- 1

      local ELz = Lms * Zms  -- p x n
      local EGx = Gms * Xms  -- p x n

      local MuDiff = Mu - (ELz * 2 + EGx * 2):view(1, p, n):expand(sn, p, n)  -- sn x p x n
      local PtMuTPsiI = torch.bmm(PsiI_M:view(1, p, p):expand(sn, p, p), torch.cmul(pPt, Mu))  -- sn x p x n

      logQs[{{}, s}] = - torch.cmul(PtMuTPsiI, MuDiff):sum(1):sum(2):view(1, n) * 0.5
               - EzTLTPsiIGx - EzTLTPsiILz * 0.5 - ExTGTPsiIGx * 0.5
               + torch.sum(torch.log(torch.diag(torch.potrf(Zcovs, 'U'))))
               + torch.sum(torch.log(torch.diag(torch.potrf(Xcovs, 'U'))))
   end

   logQs:add(cephes.digamma(phim):float():view(1, S):expand(n, S))
   logQs = logQs - torch.max(logQs, 2) * torch.ones(1, S)
   torch.exp(Qs, logQs)
   Qs:cmul(torch.sum(Qs, 2):pow(-1) * torch.ones(1, S))

   self:check(Qs, "Qs")
end


---------------------------------------------
-- Pt : n x sn
-- Mu  : sn x p x n
---------------------------------------------
function VBCMFA:inferPsiI(Pt, Mu)
   local n, S, p, k, f = self:_setandgetDims()
   local Gm, Gcov = self.factorLoading:getG()
   local Xm, Xcov = self.conditional:getX()
   local Lm, Lcov = self.factorLoading:getL()
   local Zm, Zcov = self.hidden:getZ()
   local Qs, phim = self.hidden:getS()
   local PsiI = self.hyper.PsiI

   local psi = torch.zeros(p, p)
   local sn = Pt:size(2)
   local pPt = Pt:view(1, n, sn):expand(p, n, sn):permute(3, 1, 2)  -- sn x p x n

   for s = 1, S do
      local Qss = Qs[{{}, s}]  -- n x 1
      local Zms = Zm[s]  -- k x n
      local Xms = Xm[s]  -- f x n
      local Lms = Lm[s]  -- p x k
      local Gms = Gm[s]  -- p x f

      local kQss = Qss:contiguous():view(1, n):expand(k, n)  -- k x n
      local EzzT = Zcov[s] * torch.sum(Qss) + Zms * torch.cmul(Zms, kQss):t()  -- k x k

      local dQss = Qss:contiguous():view(1, n):expand(f, n)  -- f x n
      local ExxT = Xcov[s] * torch.sum(Qss) + Xms * torch.cmul(Xms, dQss):t()  -- f x f

      local EzxT = torch.cmul(Zms, kQss) * Xms:t()  -- k x f
      local ELzxTGT = Lms * EzxT * Gms:t()  -- p x p

      local ELz = Lms * Zms  -- p x n
      local EGx = Gms * Xms  -- p x n

      local MuDiff = Mu - (ELz * 2 + EGx * 2):view(1, p, n):expand(sn, p, n)  -- sn x p x n
      local MuPt = torch.cmul(Mu, pPt)  -- sn x p x n

      local EMuPtMuDiffT = torch.bmm(MuPt, MuDiff:transpose(2, 3)):sum(1):view(p, p)  -- p x p

      local partialPsi = EMuPtMuDiffT
                       + Lms * EzzT * Lms:t()
                       + Gms * ExxT * Gms:t()
                       + ELzxTGT * 2

      psi:add(partialPsi)
      for q = 1, p do
         psi[q][q] = psi[q][q] + torch.trace(Lcov[s][q] * EzzT) + torch.trace(Gcov[s][q] * ExxT)
      end
   end

   torch.div(PsiI, torch.diag(psi), n)
   PsiI:pow(-1)
   self:check(PsiI, "PsiI")
end


---------------------------------------------
-- Pt : n x sn
-- Mu  : sn x p x n
---------------------------------------------
function VBCMFA:inferQx(Pt, Mu, X_star)
   local n, S, p, k, f = self:_setandgetDims()
   local Gm, Gcov = self.factorLoading:getG()
   local Xm, Xcov = self.conditional:getX()
   local Lm = self.factorLoading:getL()
   local Zm = self.hidden:getZ()
   local PsiI = self.hyper.PsiI
   local sigma_star = self.hyper.sigma_star

   local Sn = Pt:size(2)
   local pPt = Pt:view(n, Sn, 1):expand(n, Sn, p):permute(2, 3, 1)  -- Sn x p x n
   local PtMu = torch.cmul(pPt, Mu):sum(1):view(p, n)  -- p x n

   for s = 1, S do
      local Gcovs = Gcov[s]  -- p x f x f
      local Gms = Gm[s]  -- p x f
      local GmPsiI = Gms:t() * torch.diag(PsiI)  -- f x p

      local EGTLG = (Gcovs:view(p, f * f):t() * PsiI):view(f, f) + GmPsiI * Gms  -- f x f
      local GmPsiIPtMu = Gms:t() * torch.diag(PsiI) * (PtMu - Lm[s] * Zm[s])  -- f x n

      -- covariance
      local sigma_starI = torch.inverse(sigma_star)  -- f x f
      Xcov[s] = torch.inverse(sigma_starI + EGTLG)  -- f x f
      Xm[s] = Xcov[s] * (sigma_starI * X_star + GmPsiIPtMu)  -- f x n
   end

   self:check(Xm, "Xm")
   self:check(Xcov, "Xcov")
end


---------------------------------------------
-- To infer parameters for prior distribution
-- of feature vectors i.e. x_i
---------------------------------------------
function VBCMFA:inferHyperX()
   local n, S, p, k, f = self:_setandgetDims()

   local Xm, Xcov = self.conditional:getX()
   local mu_star, sigma_star = self.hyper.mu_star, self.hyper.sigma_star
   local Qs = self.hidden:getS()

   local mu_starT = torch.bmm(Xm:permute(3, 2, 1), Qs:view(n, S, 1))  -- n x f x 1
   mu_star = mu_starT:view(n, f):t():contiguous()  -- f x n

   local XcovQs = Xcov:view(S, f * f):t() * Qs:sum(1):t() / n  -- f*f x 1
   sigma_star = (torch.inverse(XcovQs:view(f, f))):contiguous()

   self:check(mu_star, "mu_star")
   self:check(sigma_star, "sigma_star")
end


---------------------------------------------
-- Pt : n x sn
-- Mu  : sn x p x n
---------------------------------------------
function VBCMFA:inferQG(Pt, Mu)
   local n, S, p, k, f = self:_setandgetDims()
   local Gm, Gcov, alpha, beta = self.factorLoading:getG()
   local Xm, Xcov = self.conditional:getX()
   local Lm = self.factorLoading:getL()
   local Zm = self.hidden:getZ()
   local Qs, phim = self.hidden:getS()
   local PsiI = self.hyper.PsiI

   local Sn = Pt:size(2)
   local pPt = Pt:view(n, Sn, 1):expand(n, Sn, p):permute(2, 3, 1)  -- Sn x p x n
   local PtMu = torch.cmul(pPt, Mu):sum(1):view(p, n)  -- p x n

   for s = 1, S do
      local Xcovs = Xcov[s]
      local Xms = Xm[s]
      local Qss = Qs[{{}, s}]
      local Gcovs = Gcov[s]
      local Gms = Gm[s]

      local EXmQs = torch.cmul(Xms, Qss:contiguous():view(1, n):expand(f, n))  -- f x n
      local QsXXT = Xcovs * torch.sum(Qss) + Xms * EXmQs:t()  -- f x f
      local betas = beta[s]  -- 1 x f
      local EOmega = torch.diag(betas:pow(-1) * alpha)  -- f x f

      local PsiIYQsXms = torch.diag(PsiI) * (PtMu - Lm[s] * Zm[s]) * EXmQs:t()  -- p x f


      for q = 1, p do
         torch.inverse(Gcovs[q], EOmega + QsXXT * PsiI[q])
         Gms[q] = Gcovs[q] * PsiIYQsXms[q]
      end
   end

   self:check(Gm, "Gm")
   self:check(Gcov, "Gcov")
end


---------------------------------------------
-- Pt : n x sn
-- Mu  : sn x p x n
---------------------------------------------
function VBCMFA:inferQL(Pt, Mu)
   local n, S, p, k, f = self:_setandgetDims()
   local Lm, Lcov, a, b = self.factorLoading:getL()
   local Zm, Zcov = self.hidden:getZ()
   local Gm = self.factorLoading:getG()
   local Xm = self.conditional:getX()
   local Qs = self.hidden:getS()
   local PsiI = self.hyper.PsiI

   local Sn = Pt:size(2)
   local pPt = Pt:view(n, Sn, 1):expand(n, Sn, p):permute(2, 3, 1)  -- Sn x p x n
   local PtMu = torch.cmul(pPt, Mu):sum(1):view(p, n)  -- p x n

   for s = 1, S do
      local Zcovs = Zcov[s]
      local Zms = Zm[s]
      local Qss = Qs[{{}, s}]
      local Lcovs = Lcov[s]
      local Lms = Lm[s]

      local ZmQs = torch.cmul(Zms, Qss:contiguous():view(1, n):expand(k, n)) -- k x n
      local QsZZT = Zcovs * torch.sum(Qss) + Zms * ZmQs:t()  -- k x k
      local bs = b[s]  -- k
      local Enu = torch.diag(bs:pow(-1) * a)  -- k x k

      local PsiIYQsZms = torch.diag(PsiI) * (PtMu - Gm[s] * Xm[s]) * ZmQs:t()  -- p x k

      for q = 1, p do
         torch.inverse(Lcovs[q], Enu + QsZZT * PsiI[q])
         Lms[q] = Lcovs[q] * PsiIYQsZms[q]
      end
   end

   self:check(Lm, "Lm")
   self:check(Lcov, "Lcov")
end


---------------------------------------------
-- Pt : n x sn
-- Mu  : sn x p x n
---------------------------------------------
function VBCMFA:inferQz(Pt, Mu)
   local n, S, p, k, f = self:_setandgetDims()
   local Zm, Zcov = self.hidden:getZ()
   local Lm, Lcov = self.factorLoading:getL()
   local Gm = self.factorLoading:getG()
   local Xm = self.conditional:getX()
   local PsiI = self.hyper.PsiI

   local Sn = Pt:size(2)
   local pPt = Pt:view(n, Sn, 1):expand(n, Sn, p):permute(2, 3, 1)  -- Sn x p x n
   local PtMu = torch.cmul(pPt, Mu):sum(1):view(p, n)  -- p x n

   for s = 1, S do
      local Lms = Lm[s]
      local Lcovs = Lcov[s]

      local LmTPsiI = Lms:t() * torch.diag(PsiI)

      -- covariance
      local Eql = (Lcovs:view(p, k * k):t() * PsiI):view(k, k) + LmTPsiI * Lms
      Zcov[s] = torch.inverse(torch.eye(k) + Eql)

      -- mean
      Zm[s] = Zcov[s] * Lms:t() * torch.diag(PsiI) * (PtMu - Gm[s] * Xm[s])
   end

   self:check(Zm, "Zm")
   self:check(Zcov, "Zcov")
end


function VBCMFA:inferQpi()
   local n, s, p, k, f = self:_setandgetDims()
   local Qs, phim = self.hidden:getS()

   local phi_starm = torch.ones(s) * self.hyper.phi_star / s
   phim:add(phi_starm, torch.sum(Qs))
end


function VBCMFA:inferQnu()
   local n, S, p, k, f = self:_setandgetDims()
   local Lm, Lcov, a, b = self.factorLoading:getL()

   self.factorLoading.a = self.hyper.a_star + 0.5 * p

   for s = 1, S do
      local Lms = Lm[s]  -- p x k
      local Lcovs = Lcov[s]  -- p x k x k

      local ELq = Lcovs:sum(1):view(k, k):diag():view(1, k) + torch.pow(Lms, 2):sum(1)  -- 1 x k

      b[s] = ELq:t() * 0.5 + self.hyper.b_star
   end
end


function VBCMFA:inferQomega()
   local n, S, p, k, d = self:_setandgetDims()
   local Gm, Gcov, alpha, beta = self.factorLoading:getG()

   self.factorLoading.alpha = self.hyper.alpha_star + 0.5 * p

   for s = 1, S do
      local Gms = Gm[s]  -- p x d
      local Gcovs = Gcov[s]  -- p x d x d

      local EGq = Gcovs:sum(1):view(d, d):diag():view(1, d) + torch.pow(Gms, 2):sum(1)  -- 1 x d

      beta[s] = EGq * 0.5 + self.hyper.beta_star
   end
end


---------------------------------------------
-- Pt : n x sn
-- Mu  : sn x p x n
---------------------------------------------
function VBCMFA:calcF(debug, Mu, Pt, X_star) -- p x n, f x n
   local n, s, p, k, d = self:_setandgetDims()
   local Lm, Lcov, a, b = self.factorLoading:getL()
   local Gm, Gcov, alpha, beta = self.factorLoading:getG()
   local Zm, Zcov = self.hidden:getZ()
   local Xm, Xcov = self.conditional:getX()
   local Qs, phim = self.hidden:getS()
   local a_star, b_star, alpha_star, beta_star, mu_star, sigma_star, phi_star, PsiI = self.hyper:get()

   print(string.format("Qs for first 20 samples =\n%s", self.hidden.Qs[{{1, 20}, {}}]))
   print(string.format("Number of samples for each components = \n%s", self.hidden.Qs:sum(1)))


   local Fmatrix = self.Fmatrix -- 7 x s
   local PsiI_M = torch.diag(self.hyper.PsiI)

   local Ps = torch.sum(Qs, 1)[1] -- s

   local logDetE_star = 2 * torch.sum(torch.log(torch.diag(torch.potrf(sigma_star, 'U'))))  -- change the name to sigma_star

   local X_star_Xm = X_star:view(1, d, n):expand(s, d, n) - Xm -- s x d x n
   local Qsmod = Qs:clone()
   Qsmod[Qs:eq(0)] = 1

   local digamphim = cephes.digamma(phim) -- s
   local digsumphim = cephes.digamma(torch.sum(phim)) -- 1

   Fmatrix1 = - kldirichlet(phim, torch.ones(s) * phi_star / s)

   local F_old = self.F

   local sn = Pt:size(2)
   local pPt = Pt:view(1, sn, n):expand(p, sn, n):permute(2, 1, 3)  -- sn x p x n

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
               + 2 * torch.sum(torch.log(torch.diag(torch.potrf(Lcovt[q]))))
               - (torch.diag(Lcovt[q]) + torch.pow(Lmt[q], 2)):dot(a_bt)
      end
      Fmatrixt[3] = f3 / 2

      -- Fmatrix[4]
      Fmatrixt[4] = - klgamma(torch.ones(d) * alpha, beta[t], torch.ones(d) * alpha_star, torch.ones(d) * beta_star)

      -- Fmatrix[5]
      local alpha_betat = torch.div(beta[t], alpha):pow(-1) -- k
      local f5 = cephes.digamma(alpha) * d - torch.sum(torch.log(beta[t]))

      for q = 1, p do
         f5 = f5 - d
               + 2 * torch.sum(torch.log(torch.diag(torch.potrf(Gcovt[q]))))
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
                    + 0.5 * Ps[t] * 2 * torch.sum(torch.log(torch.diag(torch.potrf(Zcovt, 'U'))))
                    - 0.5 * torch.trace(QstEzzT)

      -- Fmatrix[8]
      local fQst = Qst:contiguous():view(1, n):expand(d, n) -- d x n
      local X_star_XmtfQst = torch.cmul(X_star_Xmt, fQst) -- d x n
      local QstExxT = Xcovt * Ps[t] + X_star_Xmt * X_star_XmtfQst:t() -- d x d

      Fmatrixt[8] = 0.5 * d * torch.sum(Qst)
                    - 0.5 * Ps[t] * logDetE_star
                    + 0.5 * Ps[t] * 2 * torch.sum(torch.log(torch.diag(torch.potrf(Xcovt, 'U'))))
                    - 0.5 * torch.trace(torch.inverse(sigma_star) * QstExxT)

      -- Fmatrix[9]
      local ELTPsiIG = Lmt:t() * PsiI_M * Gmt -- k x d
      local EzTLTPsiIGx = torch.sum(torch.cmul(Zmt, ELTPsiIG * Xmt), 1) -- 1 x n

      local ELTPsiIL = torch.view(torch.view(Lcovt, p, k * k):t() * PsiI, k, k)
                           + Lmt:t() * PsiI_M * Lmt -- k x k
      local EzTLTPsiILz = torch.sum(torch.cmul(Zmt, ELTPsiIL * Zmt), 1) -- 1 x n
                           + (torch.view(ELTPsiIL, 1, k * k) * torch.view(Zcovt:t():contiguous(), k * k, 1)):squeeze() -- 1

      local EGTPsiIG = torch.view(torch.view(Gcovt, p, d * d):t() * PsiI, d, d) -- d x d
                     + Gmt:t() * PsiI_M * Gmt -- d x d
      local ExTGTPsiIGx = torch.sum(torch.cmul(Xmt, EGTPsiIG * Xmt), 1) -- 1 x n
                        + (torch.view(EGTPsiIG, 1, d * d) * torch.view(Xcovt:t():contiguous(), d * d, 1)):squeeze() -- 1

      local ELz = Lmt * Zmt
      local EGx = Gmt * Xmt

      local MuDiff = Mu - (ELz * 2 + EGx * 2):view(1, p, n):expand(sn, p, n)  -- sn x p x n
      local PtMuTPsiI = torch.bmm(PsiI_M:view(1, p, p):expand(sn, p, p), torch.cmul(pPt, Mu))  -- sn x p x n

      local E = - torch.cmul(PtMuTPsiI, MuDiff):sum(1):sum(2):view(1, n) * 0.5
                - EzTLTPsiIGx * 2 * 0.5
                - EzTLTPsiILz * 0.5
                - ExTGTPsiIGx * 0.5

      Fmatrixt[9] = torch.cmul(E, Qst):sum() - 0.5 * Ps[t] * logDet2piPsiI
   end

   print(string.format("Fmatrix = \n%s", Fmatrix))

   self.F = torch.sum(Fmatrix) + Fmatrix1
   self.dF = self.F - F_old

   return self.F, self.dF
end


---------------------------------------------
-- Pt : n x sn
-- Mu  : sn x p x n
---------------------------------------------
function VBCMFA:doBirth(parent, Mu, Pt)
   local n, s, p, k, d = self:_setandgetDims()

   self.s = self.s + 1
   local s = self.s

   local sn = Pt:size(2)

   local Lm, Lcov, _, b = self.factorLoading:getL()
   local Gm, Gcov, _, beta = self.factorLoading:getG()
   local Xm, Xcov = self.conditional:getX()
   local Zm, Zcov = self.hidden:getZ()
   local Xm, Xcov = self.conditional:getX()
   local Qs, phim = self.hidden:getS()
   local PsiI = self.hyper.PsiI

   local Lmp, Lcovp = Lm[parent], Lcov[parent]
   local Gmp, Gcovp = Gm[parent], Gcov[parent]
   local Xmp, Xcovp = Xm[parent], Xcov[parent]
   local Zmp, Zcovp = Zm[parent], Zcov[parent]
   local Qsp = Qs[{{}, parent}]


   local pPt = Pt:view(1, sn, n):expand(p, sn, n):permute(2, 1, 3)  -- sn x p x n

   local EGxs = Gmp * Xmp  -- p x n
   local MuDiff = Mu - EGxs:view(1, p, n):expand(sn, p, n)  -- sn x p x n

   local cov = Lmp * Lmp:t() + Gmp * self.hyper.sigma_star * Gmp:t() + torch.diag(PsiI:pow(-1))
   local delta0 = distributions.mvn.rnd(torch.zeros(1, p), cov)
   local delta = Gmp * self.hyper.mu_star + delta0:view(p, 1):expand(p, n)

   local PtMuDiff = torch.cmul(pPt, MuDiff):sum(1):view(p, n)
   local assign = torch.sign(torch.cmul(delta, PtMuDiff):sum(1))

   -- update Qs
   local Qss = torch.zeros(n)
   Qss[assign:eq(1)] = Qsp[assign:eq(1)] * self.hardness
   Qsp[assign:eq(1)] = Qsp[assign:eq(1)] * (1 - self.hardness)
   Qss[assign:eq(-1)] = Qsp[assign:eq(-1)] * (1 - self.hardness)
   Qsp[assign:eq(-1)] = Qsp[assign:eq(-1)] * self.hardness
   self.hidden.Qs = self.hidden.Qs:cat(Qss)

   -- phim
   self.hidden.phim = phim:cat(torch.Tensor({phim[parent]}))
   self.hidden.phim[parent] = phim[parent] / 2

   -- Lm and Lcov
   self.factorLoading.Lm = Lm:cat(Lmp:view(1, p, k), 1)
   self.factorLoading.Lcov = Lcov:cat(Lcovp:view(1, p, k, k), 1)

   -- Gm and Gcov
   self.factorLoading.Gm = Gm:cat(Gmp:view(1, p, d), 1)
   self.factorLoading.Gcov = Gcov:cat(Gcovp:view(1, p, d, d), 1)

   -- Xm and Xcov
   -- local Ginv = Gmp:t() * torch.inverse(Gmp * Gmp:t())
   -- local XmDiff = Ginv * delta
   -- Xm[parent] = Xmp - XmDiff
   -- self.conditional.Xm = Xm:cat((Xmp + XmDiff):view(1, d, n), 1)
   self.conditional.Xm = Xm:cat(Xmp:view(1, d, n), 1)
   self.conditional.Xcov = Xcov:cat(Xcovp:view(1, d, d), 1)

   -- Zm and Zcov
   self.hidden.Zm = Zm:cat(Zmp:view(1, k, n), 1)
   self.hidden.Zcov = Zcov:cat(Zcovp:view(1, k, k), 1)

   -- b and beta
   self.factorLoading.b = b:cat(torch.ones(1, k), 1)
   self.factorLoading.beta = beta:cat(torch.ones(1, d), 1)

   --Fmatrix
   self.Fmatrix = self.Fmatrix:cat(torch.zeros(9, 1), 2)
end


-- function to order candidates for birth
function VBCMFA:orderCandidates()
   local Qs = self.hidden:getS()
   local Fmatrix = self.Fmatrix

   local free_energy = - torch.cdiv(Fmatrix[{{6, 9}, {}}]:sum(1), Qs:sum(1)) - Fmatrix[{{2, 5}, {}}]:sum(1)
   local _, order = torch.sort(free_energy, 2, true)
   return order
end


-- function to save all the parameters in given file
function VBCMFA:saveWorkspace(file)
   local workspace = {
      hidden = self.hidden,
      conditional = self.conditional,
      factorLoading = self.factorLoading,
      hyper = self.hyper,
      s = self.s,
      Fmatrix = self.Fmatrix,
      Fhist = self.Fhist
   }
   torch.save(file, workspace)
end


-- function to load parameters from given file
function VBCMFA:loadWorkspace(file)
   local workspace = torch.load(file)
   self.hidden = workspace.hidden
   self.conditional = workspace.conditional
   self.factorLoading = workspace.factorLoading
   self.hyper = workspace.hyper
   self.s = workspace.s
   self.Fmatrix = workspace.Fmatrix
   self.Fhist = workspace.Fhist
end


function VBCMFA:handleBirth(Mu, Pt, X_star)
   local file = 'workspace.dat'
   local Ftarget = self:calcF(self.debug, Mu, Pt, X_star)
   local order = self:orderCandidates()
   self:saveWorkspace(file)
   self:doBirth(order[1][1], Mu, Pt)

   for i = 1, 20 do
      self:inferQz(Pt, Mu)
      self:inferQL(Pt, Mu)
      self:inferQx(Pt, Mu, X_star)
      self:inferQG(Pt, Mu)
   end

   local F = self:calcF(self.debug, Mu, Pt, X_star)
   print(string.format("--------------------------------\n"))
   print(string.format("F = %f", F))
   print(string.format("Ftagr = %f", Ftarget))
   print("Qns = ")
   print(self.hidden.Qs:sum(1))
   print(string.format("--------------------------------\n"))

   if F > Ftarget then
      print("birth successful")
   else
      print("reverting")
      self:loadWorkspace(file)
   end
end


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


-------------------------------------------------------------------------
------------------------- Helper Functions ------------------------------
-------------------------------------------------------------------------

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


return VBCMFA