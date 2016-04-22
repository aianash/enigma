local pl = require('pl.import_into')()
require 'cephes'
require 'distributions'
require 'torchx'

local Set   = require 'pl.Set'
local Utils = import 'enigma.Utils'

local kldirichlet = Utils.kldirichlet
local klgamma     = Utils.klgamma
local posdefify   = Utils.posdefify
local logdet      = Utils.logdet
local inverse     = Utils.inverse


local MultiTargetVBCMFA, parent = klazz('enigma.cmfa.MultiTargetVBCMFA', 'enigma.cmfa.VBCMFA')

function MultiTargetVBCMFA:__init(cfg)
   self.hardness = 0.5
   parent:__init(cfg)
end


---------------------------------------------
-- Pt : n x sn
-- Mu  : sn x p x n
---------------------------------------------
function MultiTargetVBCMFA:inferQs(Mu, Pt, debug)
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

      local ELTPsiIL = torch.view(torch.view(Lcovs, p, k * k):t() * PsiI, k, k)
                     + Lms:t() * PsiI_M * Lms
      local EzTLTPsiILz = torch.sum(torch.cmul(Zms, ELTPsiIL * Zms), 1) -- 1 x n
                        + (torch.view(ELTPsiIL, 1, k * k) * torch.view(Zcovs:t():contiguous(), k * k, 1)):squeeze() -- 1

      local EGTPsiIG = torch.view(torch.view(Gcovs, p, f * f):t() * PsiI, f, f) -- f x f
                     + Gms:t() * PsiI_M * Gms
      local ExTGTPsiIGx = torch.sum(torch.cmul(Xms, EGTPsiIG * Xms), 1) -- 1 x n
                        + (torch.view(EGTPsiIG, 1, f * f) * torch.view(Xcovs:t():contiguous(), f * f, 1)):squeeze() -- 1

      local ELz = Lms * Zms  -- p x n
      local EGx = Gms * Xms  -- p x n

      local MuDiff = Mu - (ELz * 2 + EGx * 2):view(1, p, n):expand(sn, p, n)  -- sn x p x n
      local PtMuTPsiI = torch.bmm(PsiI_M:view(1, p, p):expand(sn, p, p), torch.cmul(pPt, Mu))  -- sn x p x n

      logQs[{{}, s}] = - torch.cmul(PtMuTPsiI, MuDiff):sum(1):sum(2):view(1, n) * 0.5
               - EzTLTPsiIGx - EzTLTPsiILz * 0.5 - ExTGTPsiIGx * 0.5
               + logdet(Zcovs) * 0.5
               + logdet(Xcovs) * 0.5
   end

   logQs:add(cephes.digamma(phim):float():view(1, S):expand(n, S))
   logQs = logQs - torch.max(logQs, 2) * torch.ones(1, S)
   torch.exp(Qs, logQs)
   Qs:cmul(torch.sum(Qs, 2):pow(-1) * torch.ones(1, S))

   self:check(Qs, 'Qs')
end


---------------------------------------------
-- Pt : n x sn
-- Mu  : sn x p x n
---------------------------------------------
function MultiTargetVBCMFA:inferPsiI(Mu, Pt, debug)
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

      local fQss = Qss:contiguous():view(1, n):expand(f, n)  -- f x n
      local ExxT = Xcov[s] * torch.sum(Qss) + Xms * torch.cmul(Xms, fQss):t()  -- f x f

      local EzxT = Zms * torch.cmul(Xms, fQss):t()  -- k x f
      local ELzxTGT = Lms * EzxT * Gms:t()  -- p x p

      local ELz = Lms * Zms  -- p x n
      local EGx = Gms * Xms  -- p x n

      local MuDiff = Mu - (ELz * 2 + EGx * 2):view(1, p, n):expand(sn, p, n)  -- sn x p x n
      local MuPt = torch.cmul(Mu, pPt)  -- sn x p x n
      local QstMuPt = torch.cmul(MuPt, Qss:contiguous():view(n, 1, 1):expand(n, p, sn):transpose(1, 3))

      local EMuPtMuDiffT = torch.bmm(QstMuPt, MuDiff:transpose(2, 3)):sum(1):view(p, p)  -- p x p

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

   -- check PsiI for negative values
   if torch.sum(PsiI[PsiI:lt(0)]) ~= 0 then
      local PsiI_M = torch.diag(PsiI)
      posdefify(PsiI_M)
      torch.diag(PsiI, PsiI_M)
   end

   self:check(PsiI, 'PsiI')
end


function MultiTargetVBCMFA:inferQXZ(Mu, Pt, X_star, epochs, debug)
   local n, S, p, k, f = self:_setandgetDims()
   local Gm, Gcov = self.factorLoading:getG()
   local Xm, Xcov = self.conditional:getX()
   local Lm, Lcov = self.factorLoading:getL()
   local Zm, Zcov = self.hidden:getZ()
   local PsiI = self.hyper.PsiI
   local E_starI = self.hyper.E_starI

   local Sn = Pt:size(2)
   local pPt = Pt:view(n, Sn, 1):expand(n, Sn, p):permute(2, 3, 1)  -- Sn x p x n
   local PtMu = torch.cmul(pPt, Mu):sum(1):view(p, n)  -- p x n
   local E_starIX_star = E_starI * X_star

   local GmTPsiI           = torch.zeros(S, f, p)
   local LmTPsiI           = torch.zeros(S, k, p)
   local GmTPsiIPtMu       = torch.zeros(S, f, n)
   local GmTPsiILm         = torch.zeros(S, f, k)
   local LmTPsiIPtMu       = torch.zeros(S, k, n)
   local LmTPsiIGm         = torch.zeros(S, k, f)

   for s = 1, S do
      GmTPsiI[s] = Gm[s]:t() * torch.diag(PsiI)
      local EGTLG = torch.view(torch.view(Gcov[s], p, f * f):t() * PsiI, f, f) + GmTPsiI[s] * Gm[s]
      Xcov[s] = inverse(E_starI + EGTLG)

      LmTPsiI[s] = Lm[s]:t() * torch.diag(PsiI)
      local ELTGL = torch.view(torch.view(Lcov[s], p, k * k):t() * PsiI, k, k) + LmTPsiI[s] * Lm[s]
      Zcov[s] = inverse(torch.eye(k) + ELTGL)

      GmTPsiIPtMu[s] = GmTPsiI[s] * PtMu
      GmTPsiILm[s] = GmTPsiI[s] * Lm[s]

      LmTPsiIPtMu[s] = LmTPsiI[s] * PtMu
      LmTPsiIGm[s] = LmTPsiI[s] * Gm[s]
   end

   for epoch = 1, epochs do
      for s = 1, S do
         Zm[s] = Zcov[s] * (LmTPsiIPtMu[s] - LmTPsiIGm[s] * Xm[s])

         local GmTPsiIPtMuDiff = GmTPsiIPtMu[s] - GmTPsiILm[s] * Zm[s]
         Xm[s] = Xcov[s] * (E_starIX_star + GmTPsiIPtMuDiff)
      end
   end
end


---------------------------------------------
-- Pt : n x sn
-- Mu  : sn x p x n
---------------------------------------------
function MultiTargetVBCMFA:inferQX(Mu, Pt, X_star, debug)
   local n, S, p, k, f = self:_setandgetDims()
   local Gm, Gcov = self.factorLoading:getG()
   local Xm, Xcov = self.conditional:getX()
   local Lm = self.factorLoading:getL()
   local Zm = self.hidden:getZ()
   local PsiI = self.hyper.PsiI
   local E_starI = self.hyper.E_starI

   local Sn = Pt:size(2)
   local pPt = Pt:view(n, Sn, 1):expand(n, Sn, p):permute(2, 3, 1)  -- Sn x p x n
   local PtMu = torch.cmul(pPt, Mu):sum(1):view(p, n)  -- p x n

   for s = 1, S do
      local Gcovs = Gcov[s]  -- p x f x f
      local Gms = Gm[s]  -- p x f
      local GmPsiI = Gms:t() * torch.diag(PsiI)  -- f x p

      local EGTLG = torch.view(torch.view(Gcovs, p, f * f):t() * PsiI, f, f) + GmPsiI * Gms  -- f x f
      local GmPsiIPtMu = GmPsiI * (PtMu - Lm[s] * Zm[s])  -- f x n

      -- covariance
      Xcov[s] = inverse(E_starI + EGTLG)  -- f x f
      Xm[s] = Xcov[s] * (E_starI * X_star + GmPsiIPtMu)  -- f x n
   end

   self:check(Xm, 'Xm')
   self:check(Xcov, 'Xcov')
end


function MultiTargetVBCMFA:inferQLG(Mu, Pt, epochs, debug)
   local n, S, p, k, f = self:_setandgetDims()
   local Gm, Gcov, alpha, beta = self.factorLoading:getG()
   local Lm, Lcov, a, b = self.factorLoading:getL()
   local Xm, Xcov = self.conditional:getX()
   local Zm, Zcov = self.hidden:getZ()
   local Qs, phim = self.hidden:getS()
   local PsiI = self.hyper.PsiI

   local Sn = Pt:size(2)
   local pPt = Pt:view(n, Sn, 1):expand(n, Sn, p):permute(2, 3, 1)  -- Sn x p x n
   local PtMu = torch.cmul(pPt, Mu):sum(1):view(p, n)  -- p x n

   local ZmQs          = torch.zeros(S, k, n)
   local XmQs          = torch.zeros(S, f, n)
   local XmZmQsT       = torch.zeros(S, f, k)
   local ZmXmQsT       = torch.zeros(S, k, f)
   local PsiIPtMuZmQsT = torch.zeros(S, p, k)
   local PsiIPtMuXmQsT = torch.zeros(S, p, f)
   local PsiIPtMu      = torch.diag(PsiI) * PtMu

   for s = 1, S do
      local Qss = Qs[{{}, s}]

      ZmQs[s] = torch.cmul(Zm[s], Qss:contiguous():view(1, n):expand(k, n)) -- k x n
      local QsZZT = Zcov[s] * torch.sum(Qss) + Zm[s] * ZmQs[s]:t()  -- k x k

      XmQs[s] = torch.cmul(Xm[s], Qss:contiguous():view(1, n):expand(f, n))  -- f x n
      local QsXXT = Xcov[s] * torch.sum(Qss) + Xm[s] * XmQs[s]:t()  -- f x f

      local Enu = torch.div(b[s], a):pow(-1)  -- k
      local EOmega = torch.div(beta[s], alpha):pow(-1)  -- f

      for q = 1, p do
         Lcov[s][q] = inverse(torch.diag(Enu) + QsZZT * PsiI[q])
         Gcov[s][q] = inverse(torch.diag(EOmega) + QsXXT * PsiI[q])
      end

      XmZmQsT[s] = Xm[s] * ZmQs[s]:t()
      PsiIPtMuZmQsT[s] = PsiIPtMu * ZmQs[s]:t()

      ZmXmQsT[s] = Zm[s] * XmQs[s]:t()
      PsiIPtMuXmQsT[s] = PsiIPtMu * XmQs[s]:t()
   end

   for epoch = 1, epochs do
      for s = 1, S do
         local PsiIPtMudiffQsZms = PsiIPtMuZmQsT[s] - torch.diag(PsiI) * Gm[s] * XmZmQsT[s]  -- p x k
         local PsiIPtMudiffQsXms = PsiIPtMuXmQsT[s] - torch.diag(PsiI) * Lm[s] * ZmXmQsT[s]  -- p x f

         for q = 1, p do
            Lm[s][q] = Lcov[s][q] * PsiIPtMudiffQsZms[q]
            Gm[s][q] = Gcov[s][q] * PsiIPtMudiffQsXms[q]
         end
      end
   end
end


function MultiTargetVBCMFA:inferE_starI(X_star, debug) -- f x n
   local n, s, p, k, f = self:_setandgetDims()
   local Qs = self.hidden:getS()
   local Xm, Xcov = self.conditional:getX()
   local E_starI = self.hyper.E_starI
   -- local rho = self:rho()

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
   self.hyper.E_starI = inverse(E_star)
end


---------------------------------------------
-- Pt : n x sn
-- Mu  : sn x p x n
---------------------------------------------
function MultiTargetVBCMFA:inferQG(Mu, Pt, debug)
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
      local EOmega = torch.div(betas, alpha):pow(-1)  -- f

      local PsiIYQsXms = torch.diag(PsiI) * (PtMu - Lm[s] * Zm[s]) * EXmQs:t()  -- p x f


      for q = 1, p do
         Gcovs[q] = inverse(torch.diag(EOmega) + QsXXT * PsiI[q])
         Gms[q] = Gcovs[q] * PsiIYQsXms[q]
      end
   end

   self:check(Gm, 'Gm')
   self:check(Gcov, 'Gcov')
end


---------------------------------------------
-- Pt : n x sn
-- Mu  : sn x p x n
---------------------------------------------
function MultiTargetVBCMFA:inferQL(Mu, Pt, debug)
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
      local Qss = Qs[{{}, s}]

      local ZmQs = torch.cmul(Zm[s], Qss:contiguous():view(1, n):expand(k, n)) -- k x n
      local QsZZT = Zcov[s] * torch.sum(Qss) + Zm[s] * ZmQs:t()  -- k x k
      local bs = b[s]  -- k
      local Enu = torch.div(bs, a):pow(-1)  -- k x k

      local PsiIYQsZms = torch.diag(PsiI) * (PtMu - Gm[s] * Xm[s]) * ZmQs:t()  -- p x k

      for q = 1, p do
         Lcov[s][q] = inverse(torch.diag(Enu) + QsZZT * PsiI[q])
         Lm[s][q] = Lcov[s][q] * PsiIYQsZms[q]
      end
   end

   self:check(Lm, 'Lm')
   self:check(Lcov, 'Lcov')
end


---------------------------------------------
-- Pt : n x sn
-- Mu  : sn x p x n
---------------------------------------------
function MultiTargetVBCMFA:inferQZ(Mu, Pt, debug)
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
      local Eql = torch.view(torch.view(Lcovs, p, k * k):t() * PsiI, k, k) + LmTPsiI * Lms
      Zcov[s] = inverse(torch.eye(k) + Eql)

      -- mean
      Zm[s] = Zcov[s] * LmTPsiI * (PtMu - Gm[s] * Xm[s])
   end

   self:check(Zm, 'Zm')
   self:check(Zcov, 'Zcov')
end


function MultiTargetVBCMFA:inferQnu(debug)
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


function MultiTargetVBCMFA:inferQpi(debug)
   local n, s, p, k, f = self:_setandgetDims()
   local Qs, phim = self.hidden:getS()

   local phi_starm = torch.ones(s) * self.hyper.phi_star / s
   phim:add(phi_starm, torch.sum(Qs, 1):squeeze())
end




function MultiTargetVBCMFA:inferQomega(debug)
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
function MultiTargetVBCMFA:calcF(Mu, Pt, X_star, debug) -- p x n, f x n
   local n, s, p, k, d = self:_setandgetDims()
   local Lm, Lcov, a, b = self.factorLoading:getL()
   local Gm, Gcov, alpha, beta = self.factorLoading:getG()
   local Zm, Zcov = self.hidden:getZ()
   local Xm, Xcov = self.conditional:getX()
   local Qs, phim = self.hidden:getS()
   local a_star, b_star, alpha_star, beta_star, E_starI, phi_star, PsiI = self.hyper:get()

   local Fmatrix = self.Fmatrix -- 7 x s
   local PsiI_M = torch.diag(self.hyper.PsiI)

   local Ps = torch.sum(Qs, 1)[1] -- s

   local logDetE_star = - logdet(E_starI) -- change the name to sigma_star

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
               + logdet(Lcovt[q])
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
      local fQst = Qst:contiguous():view(1, n):expand(d, n) -- d x n
      local X_star_XmtfQst = torch.cmul(X_star_Xmt, fQst) -- d x n
      local QstExxT = Xcovt * Ps[t] + X_star_Xmt * X_star_XmtfQst:t() -- d x d

      Fmatrixt[8] = 0.5 * d * torch.sum(Qst)
                    - 0.5 * Ps[t] * logDetE_star
                    + 0.5 * Ps[t] * logdet(Xcovt)
                    - 0.5 * torch.trace(E_starI * QstExxT)

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

   if debug then print(string.format('Fmatrix = %s\n', Fmatrix)) end

   self.F = torch.sum(Fmatrix) + Fmatrix1
   self.dF = self.F - F_old

   return self.F, self.dF
end


---------------------------------------------
-- Pt : n x sn
-- Mu  : sn x p x n
---------------------------------------------
function MultiTargetVBCMFA:addComponent(parent, Mu, Pt, X_star)
   local n, S, p, k, d = self:_setandgetDims()

   self.S = S + 1
   local S = self.S

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
   local bp, betap = b[parent], beta[parent]


   local pPt = Pt:view(1, sn, n):expand(p, sn, n):permute(2, 1, 3)  -- sn x p x n

   local EGxs = Gmp * Xmp  -- p x n
   local MuDiff = Mu - EGxs:view(1, p, n):expand(sn, p, n)  -- sn x p x n

   local cov = Lmp * Lmp:t() + Gmp * inverse(self.hyper.E_starI) * Gmp:t() + torch.diag(PsiI:pow(-1))
   local delta0 = distributions.mvn.rnd(torch.zeros(1, p), cov)
   local delta = Gmp * X_star + delta0:view(p, 1):expand(p, n)

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
   phim[parent] = phim[parent] / 2
   self.hidden.phim = phim:cat(torch.Tensor({phim[parent]}))


   -- Lm and Lcov
   self.factorLoading.Lm = Lm:cat(Lmp:view(1, p, k), 1)
   self.factorLoading.Lcov = Lcov:cat(Lcovp:view(1, p, k, k), 1)

   -- Gm and Gcov
   self.factorLoading.Gm = Gm:cat(Gmp:view(1, p, d), 1)
   self.factorLoading.Gcov = Gcov:cat(Gcovp:view(1, p, d, d), 1)

   -- Xm and Xcov
   self.conditional.Xm = Xm:cat(Xmp:view(1, d, n), 1)
   self.conditional.Xcov = Xcov:cat(Xcovp:view(1, d, d), 1)

   -- Zm and Zcov
   self.hidden.Zm = Zm:cat(Zmp:view(1, k, n), 1)
   self.hidden.Zcov = Zcov:cat(Zcovp:view(1, k, k), 1)

   -- b and beta
   self.factorLoading.b = b:cat(bp:view(1, k), 1)
   self.factorLoading.beta = beta:cat(betap:view(1, d), 1)

   --Fmatrix
   self.Fmatrix = self.Fmatrix:cat(torch.zeros(9, 1), 2)
end


function MultiTargetVBCMFA:handleDeath()
   local n, S, p, k, d = self:_setandgetDims()
   local Qs = self.hidden:getS()
   print(Qs:sum(1))
   local comp2remove = Set(torch.find(Qs:sum(1):lt(100), 1))
   local comp2keep = Set(torch.find(Qs:sum(1):lt(100), 0))
   local numDeaths = Set.len(comp2remove)

   if numDeaths == 0 then return false end

   print(string.format('Removing following components = %s\n', comp2remove))
   self:removeComponent(torch.LongTensor(Set.values(comp2keep)))
   return true
end


function MultiTargetVBCMFA:handleBirth(Mu, Pt, X_star, parent)
   local file = 'workspace.dat'
   local Ftarget = self:calcF(Mu, Pt, X_star, true)
   self:saveWorkspace(file)
   self:addComponent(parent, Mu, Pt, X_star)
   local i = 1
   while true do
      self:learn(Mu, Pt, X_star, i, 20, 5)
      i = i + 1
      local F, dF = self:calcF(Mu, Pt, X_star, true)
      if self:converged(F, dF) then break end
   end
   local F = self:calcF(Mu, Pt, X_star, true)
   if F > Ftarget then
      print(string.format("Keeping the birth\n"))
      return true
   else  -- revert to previous state
      print(string.format("Reverting to previous state\n"))
      self:loadWorkspace(file)
      return false
   end
end


--------------------------------------------------------------
-- function to remove components from MFA
-- comp2keep: indicces of components to be kept (LongTensor)
--------------------------------------------------------------
function MultiTargetVBCMFA:removeComponent(comp2keep)
   local n, S, p, k, d = self:_setandgetDims()
   local Lm, Lcov, _, b = self.factorLoading:getL()
   local Gm, Gcov, _, beta = self.factorLoading:getG()
   local Xm, Xcov = self.conditional:getX()
   local Zm, Zcov = self.hidden:getZ()
   local Xm, Xcov = self.conditional:getX()
   local Qs, phim = self.hidden:getS()
   local PsiI = self.hyper.PsiI

   self.s = comp2keep:size(1)
   self.Fmatrix = self.Fmatrix:index(2, comp2keep)

   self.hidden.Zm = Zm:index(1, comp2keep)
   self.hidden.Zcov = Zcov:index(1, comp2keep)

   self.hidden.Qs = Qs:index(2, comp2keep)
   self.hidden.phim = phim:index(1, comp2keep)

   self.conditional.Xm = Xm:index(1, comp2keep)
   self.conditional.Xcov = Xcov:index(1, comp2keep)

   self.factorLoading.Lm = Lm:index(1, comp2keep)
   self.factorLoading.Lcov = Lcov:index(1, comp2keep)

   self.factorLoading.Gm = Gm:index(1, comp2keep)
   self.factorLoading.Gcov = Gcov:index(1, comp2keep)

   self.factorLoading.b = b:index(1, comp2keep)
   self.factorLoading.beta = beta:index(1, comp2keep)
end


return MultiTargetVBCMFA