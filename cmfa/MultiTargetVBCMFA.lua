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
   self.hardness = cfg.hardness or 0.5
   parent:__init(cfg)

   local n, S, p, k, f, N = self:_setandgetDims()
   self.GmpX_star_N = torch.zeros(p, N)
   self.PtMuDiff_N = torch.zeros(p, N)
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
   local rho = self:rho()

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

   local rhoPsiI = torch.div(torch.diag(psi), n):pow(-1) * rho
   PsiI:add(rhoPsiI, 1 - rho, PsiI)

   -- check PsiI for negative values
   if torch.sum(PsiI[PsiI:lt(0)]) ~= 0 then
      local PsiI_M = torch.diag(PsiI)
      posdefify(PsiI_M)
      torch.diag(PsiI, PsiI_M)
   end

   self:check(PsiI, 'PsiI')

   if debug then self:pr('PsiI', 'hyper', true) end
end


function MultiTargetVBCMFA:inferQXZ(Mu, Pt, X_star, epochs, debug)
   local n, S, p, k, f = self:_setandgetDims()
   local Gm, Gcov = self.factorLoading:getG()
   local Xm, Xcov = self.conditional:getX()
   local Lm, Lcov = self.factorLoading:getL()
   local Zm, Zcov = self.hidden:getZ()
   local PsiI = self.hyper.PsiI
   local E_starI = self.hyper.E_starI
   local rho = self:rho()

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
      local rhoXcovs = Xcov[s] * (1 - rho)
      local EGTLG = torch.view(torch.view(Gcov[s], p, f * f):t() * PsiI, f, f) + GmTPsiI[s] * Gm[s]
      Xcov[s]:add(rhoXcovs, rho, inverse(E_starI + EGTLG))

      LmTPsiI[s] = Lm[s]:t() * torch.diag(PsiI)
      local rhoZcovs = Zcov[s] * (1 - rho)
      local ELTGL = torch.view(torch.view(Lcov[s], p, k * k):t() * PsiI, k, k) + LmTPsiI[s] * Lm[s]
      Zcov[s]:add(rhoZcovs, rho, inverse(torch.eye(k) + ELTGL))

      GmTPsiIPtMu[s] = GmTPsiI[s] * PtMu
      GmTPsiILm[s] = GmTPsiI[s] * Lm[s]

      LmTPsiIPtMu[s] = LmTPsiI[s] * PtMu
      LmTPsiIGm[s] = LmTPsiI[s] * Gm[s]
   end

   local rhoZm = Zm * (1 - rho)
   local rhoXm = Xm * (1 - rho)
   for epoch = 1, epochs do
      for s = 1, S do
         Zm[s] = Zcov[s] * (LmTPsiIPtMu[s] - LmTPsiIGm[s] * Xm[s])

         local GmTPsiIPtMuDiff = GmTPsiIPtMu[s] - GmTPsiILm[s] * Zm[s]
         Xm[s] = Xcov[s] * (E_starIX_star + GmTPsiIPtMuDiff)
      end
   end
   Zm:add(rhoZm, rho, Zm)
   Xm:add(rhoXm, rho, Xm)

   if debug then
      self:pr('Xcov', 'conditional', true)
      self:pr('Xm', 'conditional', true)
      self:pr('Zcov', 'hidden', true)
      self:pr('Zm', 'hidden', true)
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
   local rho = self:rho()

   local Sn = Pt:size(2)
   local pPt = Pt:view(n, Sn, 1):expand(n, Sn, p):permute(2, 3, 1)  -- Sn x p x n
   local PtMu = torch.cmul(pPt, Mu):sum(1):view(p, n)  -- p x n

   for s = 1, S do
      local Gcovs = Gcov[s]  -- p x f x f
      local Gms = Gm[s]  -- p x f
      local GmPsiI = Gms:t() * torch.diag(PsiI)  -- f x p

      local EGTLG = torch.view(torch.view(Gcovs, p, f * f):t() * PsiI, f, f) + GmPsiI * Gms  -- f x f
      local Xcovs = inverse(E_starI + EGTLG)  -- f x f
      Xcov[s]:add(Xcovs * rho, 1 - rho, Xcov[s])

      local GmPsiIPtMu = GmPsiI * (PtMu - Lm[s] * Zm[s])  -- f x n
      local Xms = Xcov[s] * (E_starI * X_star + GmPsiIPtMu)  -- f x n
      Xm[s]:add(Xms * rho, 1 - rho, Xm[s])
   end

   self:check(Xm, 'Xm')
   self:check(Xcov, 'Xcov')

   if debug then
      self:pr('Xm', 'conditional', true)
      self:pr('Xcov', 'conditional', true)
   end
end


function MultiTargetVBCMFA:inferQLG(Mu, Pt, epochs, debug)
   local n, S, p, k, f = self:_setandgetDims()
   local Gm, Gcov, alpha, beta = self.factorLoading:getG()
   local Lm, Lcov, a, b = self.factorLoading:getL()
   local Xm, Xcov = self.conditional:getX()
   local Zm, Zcov = self.hidden:getZ()
   local Qs, phim = self.hidden:getS()
   local PsiI = self.hyper.PsiI
   local rho = self:rho()

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
         Lcovsq = inverse(torch.diag(Enu) + QsZZT * PsiI[q])
         Gcovsq = inverse(torch.diag(EOmega) + QsXXT * PsiI[q])

         Lcov[s][q]:add(Lcovsq * rho, 1 - rho, Lcov[s][q])
         Gcov[s][q]:add(Gcovsq * rho, 1 - rho, Gcov[s][q])
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
            Lmsq = Lcov[s][q] * PsiIPtMudiffQsZms[q]
            Gmsq = Gcov[s][q] * PsiIPtMudiffQsXms[q]

            Lm[s][q]:add(Lmsq * rho, 1 - rho, Lm[s][q])
            Gm[s][q]:add(Gmsq * rho, 1 - rho, Gm[s][q])
         end
      end
   end

   if debug then
      self:pr('Lm', 'factorLoading', true)
      self:pr('Lcov', 'factorLoading', true)
      self:pr('Gm', 'factorLoading', true)
      self:pr('Gcov', 'factorLoading', true)
   end
end


function MultiTargetVBCMFA:inferE_starI(X_star, debug) -- f x n
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

   E_starI:add(inverse(E_star) * rho, 1 - rho, E_starI)

   if debug then self:pr('E_starI', 'hyper', true) end
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
   local rho = self:rho()

   local Sn = Pt:size(2)
   local pPt = Pt:view(n, Sn, 1):expand(n, Sn, p):permute(2, 3, 1)  -- Sn x p x n
   local PtMu = torch.cmul(pPt, Mu):sum(1):view(p, n)  -- p x n

   for s = 1, S do
      local Qss = Qs[{{}, s}]

      local EXmQs = torch.cmul(Xm[s], Qss:contiguous():view(1, n):expand(f, n))  -- f x n
      local QsXXT = Xcov[s] * torch.sum(Qss) + Xm[s] * EXmQs:t()  -- f x f
      local betas = beta[s]  -- 1 x f
      local EOmega = torch.div(betas, alpha):pow(-1)  -- f

      local PsiIYQsXms = torch.diag(PsiI) * (PtMu - Lm[s] * Zm[s]) * EXmQs:t()  -- p x f

      for q = 1, p do
         Gcovsq = inverse(torch.diag(EOmega) + QsXXT * PsiI[q])
         Gcov[s][q]:add(Gcovsq * rho, 1 - rho, Gcov[s][q])

         Gmsq = Gcov[s][q] * PsiIYQsXms[q]
         Gm[s][q]:add(Gmsq * rho, 1 - rho, Gm[s][q])
      end
   end

   self:check(Gm, 'Gm')
   self:check(Gcov, 'Gcov')

   if debug then
      self:pr('Gm', 'factorLoading', true)
      self:pr('Gcov', 'factorLoading', true)
   end
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
   local rho = self:rho()

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
         Lcovsq = inverse(torch.diag(Enu) + QsZZT * PsiI[q])
         Lcov[s][q]:add(Lcovsq * rho, 1 - rho, Lcov[s][q])

         Lmsq = Lcov[s][q] * PsiIYQsZms[q]
         Lm[s][q]:add(Lmsq * rho, 1 - rho, Lm[s][q])
      end
   end

   self:check(Lm, 'Lm')
   self:check(Lcov, 'Lcov')

   if debug then
      self:pr('Lm', 'factorLoading', true)
      self:pr('Lcov', 'factorLoading', true)
   end
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
   local rho = self:rho()

   local Sn = Pt:size(2)
   local pPt = Pt:view(n, Sn, 1):expand(n, Sn, p):permute(2, 3, 1)  -- Sn x p x n
   local PtMu = torch.cmul(pPt, Mu):sum(1):view(p, n)  -- p x n

   for s = 1, S do
      local Lms = Lm[s]
      local Lcovs = Lcov[s]

      local LmTPsiI = Lms:t() * torch.diag(PsiI)

      -- covariance
      local Eql = torch.view(torch.view(Lcovs, p, k * k):t() * PsiI, k, k) + LmTPsiI * Lms
      Zcovs = inverse(torch.eye(k) + Eql)
      Zcov[s]:add(Zcovs * rho, 1 - rho, Zcov[s])

      -- mean
      Zms = Zcov[s] * LmTPsiI * (PtMu - Gm[s] * Xm[s])
      Zm[s]:add(Zms * rho, 1 - rho, Zm[s])
   end

   self:check(Zm, 'Zm')
   self:check(Zcov, 'Zcov')

   if debug then
      self:pr('Zm', 'hidden', true)
      self:pr('Zcov', 'hidden', true)
   end
end


function MultiTargetVBCMFA:inferQnu(debug)
   local n, S, p, k, f = self:_setandgetDims()
   local Lm, Lcov, a, b = self.factorLoading:getL()
   local rho = self:rho()

   self.factorLoading.a = self.hyper.a_star + 0.5 * p

   for s = 1, S do
      local ELq = Lcov[s]:sum(1):view(k, k):diag():view(1, k) + torch.pow(Lm[s], 2):sum(1)  -- 1 x k
      local bs = ELq:t() * 0.5 + self.hyper.b_star
      b[s] = bs * rho + b[s] * (1 - rho)
   end

   if debug then
      self:pr('a', 'factorLoading', true)
      self:pr('b', 'factorLoading', true)
   end
end


function MultiTargetVBCMFA:inferQpi(debug)
   local n, s, p, k, f = self:_setandgetDims()
   local Qs, phim = self.hidden:getS()
   local rho = self:rho()

   local phi_starm = torch.ones(s) * self.hyper.phi_star / s
   local rhophim = torch.add(phi_starm, torch.sum(Qs, 1):squeeze()) * rho
   phim:add(rhophim, 1 - rho, phim)

   if debug then self:pr('phim', 'hidden', true) end
end




function MultiTargetVBCMFA:inferQomega(debug)
   local n, S, p, k, d = self:_setandgetDims()
   local Gm, Gcov, alpha, beta = self.factorLoading:getG()
   local rho = self:rho()

   self.factorLoading.alpha = self.hyper.alpha_star + 0.5 * p

   for s = 1, S do
      local EGq = Gcov[s]:sum(1):view(d, d):diag():view(1, d) + torch.pow(Gm[s], 2):sum(1)  -- 1 x d
      local betas = EGq * 0.5 + self.hyper.beta_star
      beta[s] = betas * rho + beta[s] * (1 - rho)
   end

   if debug then
      self:pr('alpha', 'factorLoading', true)
      self:pr('beta', 'factorLoading', true)
   end
end


function MultiTargetVBCMFA:calcFbatch(Mu, Pt, X_star)
   local n, S, p, k, f, N = self:_setandgetDims()
   local Lm, Lcov = self.factorLoading:getL()
   local Gm, Gcov = self.factorLoading:getG()
   local Zm, Zcov = self.hidden:getZ()
   local Xm, Xcov = self.conditional:getX()
   local Qs, phim = self.hidden:getS()
   local a_star, b_star, alpha_star, beta_star, E_starI, phi_star, PsiI = self.hyper:get()

   local Fmatrix = self.Fmatrix
   local PsiI_M = torch.diag(PsiI)

   local Ps = torch.sum(Qs, 1)[1]
   local X_star_Xm = X_star:view(1, f, n):expand(S, f, n) - Xm -- s x d x n
   local Qsmod = Qs:clone()
   Qsmod[Qs:eq(0)] = 1

   local logDetE_star = - logdet(E_starI)

   local digamphim = cephes.digamma(phim) -- s
   local digsumphim = cephes.digamma(torch.sum(phim)) -- 1

   local sn = Pt:size(2)
   local pPt = Pt:view(1, sn, n):expand(p, sn, n):permute(2, 1, 3)  -- sn x p x n

   for s = 1, S do
      local Lms, Lcovs = Lm[s], Lcov[s]
      local Gms, Gcovs = Gm[s], Gcov[s]
      local Xms, Xcovs = Xm[s], Xcov[s]
      local Zms, Zcovs = Zm[s], Zcov[s]
      local Qss = Qs[{{}, s}] -- n
      local Qsmods = Qsmod[{{}, s}] -- n
      local X_star_Xms = X_star_Xm[s] -- f x n
      local Fmatrixs = Fmatrix[{{}, s}]

      local logDet2piPsiI = - torch.log(PsiI):sum() + p * math.log(2 * math.pi) -- 1

      Fmatrixs[6] = Fmatrixs[6] + torch.sum(torch.cmul(Qss, - torch.log(Qsmods) + torch.ones(n) * (digamphim[s] - digsumphim)))

      -- Fmatrix[7]
      local kQss = Qss:contiguous():view(1, n):expand(k, n) -- k x n
      local ZmtkQss = torch.cmul(Zms, kQss) -- k x n
      local QssEzzT = Zcovs * Ps[s] + Zms * ZmtkQss:t()
      Fmatrixs[7] = Fmatrixs[7]
                  + 0.5 * k * torch.sum(Qss)
                  + 0.5 * Ps[s] * logdet(Zcovs)
                  - 0.5 * torch.trace(QssEzzT)

      -- Fmatrix[8]
      local fQss = Qss:contiguous():view(1, n):expand(f, n) -- f x n
      local X_star_XmsfQss = torch.cmul(X_star_Xms, fQss) -- f x n
      local QssExxT = Xcovs * Ps[s] + X_star_Xms * X_star_XmsfQss:t()
      Fmatrixs[8] = Fmatrixs[8]
                  + 0.5 * f * torch.sum(Qss)
                  - 0.5 * Ps[s] * logDetE_star
                  + 0.5 * Ps[s] * logdet(Xcovs)
                  - 0.5 * torch.trace(E_starI * QssExxT)

      -- Fmatrix[9]
      local ELTPsiIG = Lms:t() * PsiI_M * Gms -- k x f
      local EzTLTPsiIGx = torch.sum(torch.cmul(Zms, ELTPsiIG * Xms), 1) -- 1 x n

      local ELTPsiIL = torch.view(torch.view(Lcovs, p, k * k):t() * PsiI, k, k)
                           + Lms:t() * PsiI_M * Lms -- k x k
      local EzTLTPsiILz = torch.sum(torch.cmul(Zms, ELTPsiIL * Zms), 1) -- 1 x n
                           + (torch.view(ELTPsiIL, 1, k * k) * torch.view(Zcovs:t():contiguous(), k * k, 1)):squeeze() -- 1

      local EGTPsiIG = torch.view(torch.view(Gcovs, p, f * f):t() * PsiI, f, f) -- f x f
                     + Gms:t() * PsiI_M * Gms -- f x f
      local ExTGTPsiIGx = torch.sum(torch.cmul(Xms, EGTPsiIG * Xms), 1) -- 1 x n
                        + (torch.view(EGTPsiIG, 1, f * f) * torch.view(Xcovs:t():contiguous(), f * f, 1)):squeeze() -- 1

      local ELz = Lms * Zms
      local EGx = Gms * Xms

      local MuDiff = Mu - (ELz * 2 + EGx * 2):view(1, p, n):expand(sn, p, n)  -- sn x p x n
      local PtMuTPsiI = torch.bmm(PsiI_M:view(1, p, p):expand(sn, p, p), torch.cmul(pPt, Mu))  -- sn x p x n

      local E = - torch.cmul(PtMuTPsiI, MuDiff):sum(1):sum(2):view(1, n) * 0.5
                - EzTLTPsiIGx * 2 * 0.5
                - EzTLTPsiILz * 0.5
                - ExTGTPsiIGx * 0.5

      Fmatrixs[9] = Fmatrixs[9] + torch.cmul(E, Qss):sum() - 0.5 * Ps[s] * logDet2piPsiI
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

   local logDetE_star = - logdet(E_starI)

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
   end

   if debug then print(string.format('Fmatrix = %s\n', Fmatrix)) end

   self.F = torch.sum(Fmatrix) + Fmatrix1
   self.dF = self.F - F_old

   return self.F, self.dF
end


function MultiTargetVBCMFA:resetF()
   self.Fmatrix = torch.zeros(9, self.S)
end


function MultiTargetVBCMFA:computeRandomStuff(Mu, Pt, X_star, batchperm)
   local n, S, p, k, d, N = self:_setandgetDims()
   local sn = Pt:size(2)

   local pPt = Pt:view(1, sn, n):expand(p, sn, n):permute(2, 1, 3)
   local EGxs = Gmp * Xmp
   local MuDiff = Mu - EGxs:view(1, p, n):expand(sn, p, n)
   local PtMuDiff = torch.cmul(pPt, MuDiff):sum(1):view(p, n)

   self.PtMuDiff_N:indexCopy(2, batchperm, PtMuDiff)
   self.GmpX_star_N:indexCopy(2, batchperm, Gmp * X_star)
end


---------------------------------------------
-- Pt : n x sn
-- Mu  : sn x p x n
---------------------------------------------
function MultiTargetVBCMFA:addComponent(parent, Mu, Pt, X_star)
   local n, S, p, k, d, N = self:_setandgetDims()

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

   local cov = Lmp * Lmp:t() + Gmp * inverse(self.hyper.E_starI) * Gmp:t() + torch.diag(PsiI:pow(-1))
   local delta0 = distributions.mvn.rnd(torch.zeros(1, p), cov)
   local delta = self.GmpX_star_N + delta0:view(p, 1):expand(p, N)
   local assign = torch.sign(torch.cmul(delta, self.PtMuDiff_N):sum(1))

   -- update Qs
   local Qss = torch.zeros(N)
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
   self:saveWorkspace(file)

   print(string.format('Adding component for parent = %d', parent))
   self:addComponent(parent, Mu, Pt, X_star)
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

   self.S = comp2keep:size(1)
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