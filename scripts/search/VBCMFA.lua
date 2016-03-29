require 'torch'
require 'cephes'

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
      Zcov = torch.zeros(s, k, k),

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
      Xcov = torch.eye(d, d):repeatTensor(s, n, 1, 1),

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
      mu_star = torch.ones(d, n) / d,
      sigma_star = torch.eye(d, d):repeatTensor(n, 1, 1),

      a_star = 1,
      b_star = 1,

      alpha_star = 1,
      beta_star = 1,

      phi_star = 1,

      PsiI = torch.ones(p) / p,

      get = function(self)
         return self.a_star, b_star, alpha_star, beta_star, self.mu_star, self.sigma_star, self.phi_star, self.PsiI
      end
   }
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
-- Pti : n x sn
-- Mu  : sn x p x n
---------------------------------------------
function VBCMFA:inferQs(Pti, Mu)
   local n, S, p, k, d = self:_setandgetDims()
   local Gm, Gcov = self.factorLoading:getG()
   local Xm, Xcov = self.conditional:getX()
   local Lm, Lcov = self.factorLoading:getL()
   local Zm, Zcov = self.hidden:getZ()
   local Qs, phim = self.hidden:getS()
   local PsiI = self.hyper.PsiI

   local sn = Pti:size(2)
   local pPti = torch.view(Pti, 1, sn, n):expand(p, sn, n):permute(2, 1, 3)  -- sn x p x n

   local logQs = torch.zeros(n, S) -- n x S

   for s = 1, S do
      local Xms = Xm[s]  -- d x n
      local Zms = Zm[s]  -- k x n
      local Lms = Lm[s]  -- p x k
      local Gms = Gm[s]  -- p x d
      local Xcovs = Xcov[s]  -- n x d x d
      local Zcovs = Zcov[s]  -- k x k
      local Lcovs = Lcov[s]  -- p x k x k
      local Gcovs = Gcov[s]  -- p x d x d
      local PsiI_M = torch.diag(PsiI) -- p x p
      local logQss = logQs[{{}, s}]  -- n x 1

      local ELTPsiIG = Lms:t() * PsiI_M * Gms  -- k x d
      local EzTLTPsiIGx = torch.sum(torch.cmul(Zms, ELTPsiIG * Xms), 1)  -- 1 x n

      local ELTPsiIL = torch.view(torch.view(Lcovs, p, k * k):t() * PsiI, k, k)
                     + Lms:t() * PsiI_M * Lms
      local EzTLTPsiILz = torch.sum(torch.cmul(Zms, ELTPsiIL * Zms), 1) -- 1 x n
                        + (torch.view(ELTPsiIL, 1, k * k) * torch.view(Zcovs:t():contiguous(), k * k, 1)):squeeze() -- 1

      local EGTPsiIG = torch.view(torch.view(Gcovs, p, d * d):t() * PsiI, d, d) -- d x d
                     + Gms:t() * PsiI_M * Gms
      local ExTGTPsiIGx = torch.sum(torch.cmul(Xms, EGTPsiIG * Xms), 1) -- 1 x n
                        + torch.view(Xcovs, n, d * d) * torch.view(EGTPsiIG:t():contiguous(), d * d, 1) -- n x 1

      local ELz = Lms * Zms  -- p x n
      local EGx = Gms * Xms  -- p x n

      local MuDiff = Mu - (ELz * 2 + EGx * 2):view(1, p, n):expand(sn, p, n)  -- sn x p x n
      local PtiMuTPsiI = torch.bmm(PsiI_M:view(1, p, p):expand(sn, p, p), torch.cmul(pPti, Mu))  -- sn x p x n

      for j = 1, n do
         logQss[j] = torch.sum(torch.log(torch.diag(torch.potrf(Xcovs[j], 'U'))))
      end

      logQss:add( - torch.cmul(PtiMuTPsiI, MuDiff):sum(1):sum(2):view(1, n) * 0.5
                  - EzTLTPsiIGx - EzTLTPsiILz * 0.5 - ExTGTPsiIGx * 0.5
                  + torch.sum(torch.log(torch.diag(torch.potrf(Zcovs, 'U')))))
   end

   logQs:add(cephes.digamma(phim):float():view(1, S):expand(n, S))
   logQs = logQs - torch.max(logQs, 2) * torch.ones(1, S)
   torch.exp(Qs, logQs)
   Qs:cmul(torch.sum(Qs, 2):pow(-1) * torch.ones(1, S))
end


---------------------------------------------
-- Pti : n x sn
-- Mu  : sn x p x n
---------------------------------------------
function VBCMFA:inferPsiI(Pti, Mu)
   local n, s, p, k, d = self:_setandgetDims()
   local Gm, Gcov = self.factorLoading:getG()
   local Xm, Xcov = self.conditional:getX()
   local Lm, Lcov = self.factorLoading:getL()
   local Zm, Zcov = self.hidden:getZ()
   local Qs, phim = self.hidden:getS()
   local PsiI = self.hyper.PsiI

   local EMu = torch.zeros(p, n)
   local EMuMuT = torch.zeros(p, p)

   local psi = torch.zeros(p, p)
   local sn = Pti:size(2)

   for i = 1, sn do
      local resp = Pti[{{}, i}]  -- n x 1
      local Mui = Mu[i]  -- p x n
      local respMui = torch.cmul(resp:contiguous():view(1, n):repeatTensor(p, 1), Mui)  -- p x n
      local MuMuT_N = respMui * Mui:t()  -- p x p

      EMuMuT = EMuMuT + MuMuT_N  -- p x p
      EMu = EMu + respMui  -- p x n
   end

   for i = 1, s do
      local Qsi = Qs[{ {}, i }]  -- n x 1
      local Zmi = Zm[i]  -- k x n
      local Xmi = Xm[i]  -- d x n
      local Xcovi = Xcov[i] -- n x d x d
      local Lmi = Lm[i]  -- p x k
      local Gmi = Gm[i]  -- p x d

      local kQsi = Qsi:repeatTensor(k, 1)  -- k x n
      local EzzT = Zcov[i] * torch.sum(Qsi) + Zmi * torch.cmul(Zmi, kQsi):t()  -- k x k

      local dQsi = Qsi:repeatTensor(d, 1)  -- d x n
      local EXcoviQs = torch.view(torch.view(Xcovi, n, d * d):t() * Qsi, d, d)  -- d x d
      local ExxT = EXcoviQs + Xmi * torch.cmul(Xmi, dQsi):t()  -- d x d

      local EzxT = torch.cmul(Zmi, kQsi) * Xmi:t()  -- k x d
      local ELzxTGT = Lmi * EzxT * Gmi:t()  -- p x p

      local partialPsi = EMuMuT - EMu * (Zmi:t() * Lmi:t() - Xmi:t() * Gmi:t()) * 2
                       + Lmi * EzzT * Lmi:t()
                       + Gmi * ExxT * Gmi:t()
                       + ELzxTGT * 2

      psi:add(partialPsi)
      for q = 1, p do
         psi[q][q] = psi[q][q] + torch.trace(Lcov[i][q] * EzzT) + torch.trace(Gcov[i][q] * ExxT)
      end
   end

   torch.div(PsiI, torch.diag(psi), n)
   PsiI:pow(-1)
end


---------------------------------------------
-- Pti : n x sn
-- Mu  : sn x p x n
---------------------------------------------
function VBCMFA:inferQx(Pti, Mu)
   local n, S, p, k, d = self:_setandgetDims()
   local Gm, Gcov = self.factorLoading:getG()
   local Xm, Xcov = self.conditional:getX()
   local Lm = self.factorLoading:getL()
   local Zm = self.hidden:getZ()
   local PsiI = self.hyper.PsiI
   local mu_star, sigma_star = self.hyper.mu_star, self.hyper.sigma_star

   local Sn = Pti:size(2)
   local PtiMu = torch.Tensor(p, n):fill(0)

   for sn = 1, Sn do
      local resp = Pti[{{}, sn}] -- n x 1
      local mean = Mu[sn] -- p x n
      PtiMu = PtiMu + mean * torch.diag(resp) -- p x n
   end

   for s = 1, S do
      local Gcovs = Gcov[s]  -- p x d x d
      local Gms = Gm[s]  -- p x d
      local GmPsiI = Gms:t() * torch.diag(PsiI)  -- d x p
      local Xcovs = Xcov[s]  -- n x d x d
      local Xms = Xm[s]  -- d x n

      local EGTLG = torch.reshape(Gcovs:reshape(p, d * d):t() * PsiI, d, d) + GmPsiI * Gms  -- d x d
      local GmPsiIPtiMu = Gms:t() * torch.diag(PsiI) * (PtiMu - Lm[s] * Zm[s])  -- d x n

      for j = 1, n do
         local sigma_starjI = torch.inverse(sigma_star[j])  -- d x d
         Xcovs[j] = torch.inverse(sigma_starjI + EGTLG)  -- d x d
         Xms[{{}, j}] = Xcovs[j] * (sigma_starjI * mu_star[{{}, j}] + GmPsiIPtiMu[{{}, j}])  -- d x 1
      end
   end
end


---------------------------------------------
-- To infer parameters for prior distribution
-- of feature vectors i.e. x_i
---------------------------------------------
function VBCMFA:inferHyperX()
   local n, S, p, k, d = self:_setandgetDims()

   local Xm, Xcov = self.conditional:getX()
   local mu_star, sigma_star = self.hyper.mu_star, self.hyper.sigma_star
   local Qs = self.hidden:getS()

   local mu_starT = torch.bmm(Xm:permute(3, 2, 1), Qs:reshape(n, S, 1))  -- n x d x 1
   mu_star = mu_starT:reshape(n, d):t()  -- d x n

   local E_XXT = torch.bmm(Xcov:permute(2, 1, 3, 4):reshape(n, S, d * d):permute(1, 3, 2), Qs:reshape(n, S, 1)):reshape(n, d, d)  -- n x d x d
   local E_XMUT = torch.bmm(mu_starT, mu_starT:permute(1, 3, 2))  -- n x d x d

   sigma_star = E_XXT - E_XMUT  -- n x d x d

   for i = 1, n do
      torch.inverse(sigma_star[i], sigma_star[i])
   end
end


---------------------------------------------
-- Pti : n x sn
-- Mu  : sn x p x n
---------------------------------------------
function VBCMFA:inferQG(Pti, Mu)
   local n, S, p, k, d = self:_setandgetDims()
   local Gm, Gcov, alpha, beta = self.factorLoading:getG()
   local Xm, Xcov = self.conditional:getX()
   local Lm = self.factorLoading:getL()
   local Zm = self.hidden:getZ()
   local Qs, phim = self.hidden:getS()
   local PsiI = self.hyper.PsiI

   local Sn = Pti:size(2) -- Pti => n x Sn
   local PtiMu = torch.Tensor(p, n):fill(0)
   for sn = 1, Sn do
      local resp = Pti[{{}, sn}] -- n x 1
      local mean = Mu[sn] -- p x n
      PtiMu = PtiMu + mean * torch.diag(resp) -- p x n
   end

   for s = 1, S do
      local Xcovs = Xcov[s]
      local Xms = Xm[s]
      local Qss = Qs[{{}, s}]
      local Gcovs = Gcov[s]
      local Gms = Gm[s]

      local EXmQs = torch.cmul(Xms, torch.repeatTensor(Qss:contiguous():view(1, n), k, 1)) -- d x n
      local EXcovQs = torch.reshape(torch.reshape(Xcovs, n, d * d):t() * Qss, d, d) -- d x d
      local QsXXT = EXcovQs + Xms * EXmQs:t()
      local betas = beta[s]
      local EOmega = torch.diag(betas:pow(-1) * alpha)
      local PsiIYQsXmi = torch.diag(PsiI) * (PtiMu - Lm[s] * Zm[s]) * EXmQs:t()
      for j = 1, p do
         torch.inverse(Gcovs[j], EOmega + QsXXT * PsiI[j])
         Gms[j] = Gcovs[j] * PsiIYQsXmi[j]
      end
   end
end


---------------------------------------------
-- Pti : n x sn
-- Mu  : sn x p x n
---------------------------------------------
function VBCMFA:inferQL(Pti, Mu)
   local n, S, p, k, d = self:_setandgetDims()
   local Lm, Lcov, a, b = self.factorLoading:getL()
   local Zm, Zcov = self.hidden:getZ()
   local Gm = self.factorLoading:getG()
   local Xm = self.conditional:getX()
   local Qs = self.hidden:getS()
   local PsiI = self.hyper.PsiI

   local Sn = Pti:size(2) -- Pti => n x Sn
   local PtiMu = torch.Tensor(p, n):fill(0)
   for sn = 1, Sn do
      local resp = Pti[{{}, sn}] -- n x 1
      local mean = Mu[sn] -- p x n
      PtiMu = PtiMu + mean * torch.diag(resp) -- p x n
   end

   for s = 1, S do
      local Zcovs = Zcov[s]
      local Zms = Zm[s]
      local Qss = Qs[{{}, s}]
      local Lcovs = Lcov[s]
      local Lms = Lm[s]

      local ZmQs = torch.cmul(Zms, torch.repeatTensor(Qss:contiguous():view(1, n), k, 1)) -- k x n
      local QsZZT = Zcovs * torch.sum(Qss) + Zms * ZmQs:t()
      local bs = b[s]
      local Enu = torch.diag(bs:pow(-1) * a)
      local PsiIYQsZmi = torch.diag(PsiI) * (PtiMu - Gm[s] * Xm[s]) * ZmQs:t()
      for j = 1, p do
         torch.inverse(Lcovs[j], Enu + QsZZT * PsiI[j])
         Lms[j] = Lcovs[j] * PsiIYQsZmi[j]
      end
   end
end


---------------------------------------------
-- Pti : n x sn
-- Mu  : sn x p x n
---------------------------------------------
function VBCMFA:inferQz(Pti, Mu)
   local n, S, p, k, d = self:_setandgetDims()
   local Zm, Zcov = self.hidden:getZ()
   local Lm, Lcov = self.factorLoading:getL()
   local Gm = self.factorLoading:getG()
   local Xm = self.conditional:getX()
   local PsiI = self.hyper.PsiI

   local Sn = Pti:size(2) -- Pti => n x Sn
   local PtiMu = torch.Tensor(p, n):fill(0)
   for sn = 1, Sn do
      local resp = Pti[{{}, sn}] -- n x 1
      local mean = Mu[sn] -- p x n
      PtiMu = PtiMu + mean * torch.diag(resp) -- p x n
   end

   for s = 1, S do
      local Lms = Lm[s]
      local Zcovs = Zcov[s]

      local LmTPsiI = Lms:t() * torch.diag(PsiI)

      -- covariance
      local Eql = torch.reshape(torch.reshape(Lcov[s], p, k * k):t() * PsiI, k, k) + LmTPsiI * Lms
      torch.inverse(Zcovs, torch.eye(k) + Eql)

      -- mean
      Zm[s] = Zcovs * Lms:t() * torch.diag(PsiI) * (PtiMu - Gm[s] * Xm[s])
   end
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
   local n, S, p, k, f = self:_setandgetDims()
   local Gm, Gcov, alpha, beta = self.factorLoading:getG()

   self.factorLoading.alpha = self.hyper.alpha_star + 0.5 * p

   for s = 1, S do
      local Gms = Gm[s]  -- p x d
      local Gcovs = Gcov[s]  -- p x d x d

      local EGq = Gcovs:sum(1):view(d, d):diag():view(1, d) + torch.pow(Gms, 2):sum(1)  -- 1 x d

      beta[s] = EGq * 0.5 + self.hyper.beta_star
   end
end

return VBCMFA