local pl = (require 'pl.import_into')()

local Utils = import 'enigma.Utils'
local inverse = Utils.inverse

local CMFA, parent = klazz('enigma.scripts.search.cmfa', 'enigma.cmfa.MultiTargetVBCMFA')

function CMFA:__init(cfg)
   parent:__init(cfg)

   local n, S, p, k, f, N = self:_setandgetDims()

   self.Xm_N = torch.zeros(S, f, N)
   self.Zm_N = torch.randn(S, k, N)
   self.Qs_N = torch.ones(N, S) / S

   self.inbirthprocess = false
   self.lastbirthsuccess = true
   self.deathstatus = true
   self.order = torch.Tensor(0)
   self.candidx = 0

   self.Fold = math.huge
   self.GmpX_star_N = torch.zeros(S, p, N)
   self.PtMuDiff_N = torch.zeros(S, p, N)
end


function CMFA:batchtrain(Ybatch, X_starbatch, batchperm, batchIdx, epochs)
   local n, s, p, k, f, N = self:_setandgetDims()
   -- assert(self:_checkDimensions(Ybatch, p, n))
   -- assert(self:_checkDimensions(X_starbatch, f, n))

   print(string.format([[
----------------------
CMFA training batch %d
----------------------
]], batchIdx))

   self.conditional.Xm:copy(X_starbatch:repeatTensor(s, 1, 1))
   self.hidden.Qs:index(self.Qs_N, 1, batchperm)

   self:rho(batchIdx)
   print(string.format('rho = %f\n', self:rho()))

   local pause = false
   local debug = false

   local Mubatch = Ybatch.Mu
   local Ptbatch = Ybatch.Pt

   for epoch = 1, epochs do
      self:infer('QXZ', pause, Mubatch, Ptbatch, X_starbatch, 10, debug)
      self:infer('QLG', pause, Mubatch, Ptbatch, 10, debug)

      for subEpoch = 1, 10 do
         self:infer('Qnu', pause, debug)
         self:infer('Qomega', pause, debug)
      end

      self:infer('Qs', pause, Mubatch, Ptbatch, debug)
      self:infer('Qpi', pause, debug)

      if epoch % 2 == 0 then
         for subEpoch = 1, 5 do
            self:infer('PsiI', pause, Mubatch, Ptbatch, debug)
            self:infer('E_starI', pause, X_starbatch, debug)
         end
      end
      xlua.progress(epoch, epochs)
   end
   xlua.progress(epochs, epochs)


   self:calcFbatch(Mubatch, Ptbatch, X_starbatch)
   self:computeBirthFactors(Mubatch, Ptbatch, X_starbatch, batchperm)

   self.Xm_N:indexCopy(3, batchperm, self.conditional.Xm)
   self.Zm_N:indexCopy(3, batchperm, self.hidden.Zm)
   self.Qs_N:indexCopy(1, batchperm, self.hidden.Qs)

   return F
end


function CMFA:computeBirthFactors(Mu, Pt, X_star, batchperm)
   local n, S, p, k, d, N = self:_setandgetDims()
   local Gm = self.factorLoading:getG()
   local Xm = self.conditional:getX()

   local sn = Pt:size(2)
   local pPt = Pt:view(1, sn, n):expand(p, sn, n):permute(2, 1, 3)

   for s = 1, S do
      local EGxs = Gm[s] * Xm[s]
      local MuDiff = Mu - EGxs:view(1, p, n):expand(sn, p, n)
      local PtMuDiff = torch.cmul(pPt, MuDiff):sum(1):view(p, n)
      self.PtMuDiff_N[s]:indexCopy(2, batchperm, PtMuDiff)
      self.GmpX_star_N[s]:indexCopy(2, batchperm, Gm[s] * X_star)
   end
end


function CMFA:addComponent(parentcomp)
   local n, S, p, k, f, N = self:_setandgetDims()
   local Lm = self.factorLoading:getL()
   local Gm = self.factorLoading:getG()
   local PsiI = self.hyper.PsiI
   local Qsp = self.Qs_N[{{}, parentcomp}]

   self.Xm_N = self.Xm_N:cat(torch.zeros(1, f, N), 1)
   self.Zm_N = self.Zm_N:cat(torch.zeros(1, k, N), 1)
   self.GmpX_star_N = self.GmpX_star_N:cat(torch.zeros(1, p, N), 1)
   self.PtMuDiff_N = self.PtMuDiff_N:cat(torch.zeros(1, p, N), 1)

   local cov = Lm[parentcomp] * Lm[parentcomp]:t() + Gm[parentcomp] * inverse(self.hyper.E_starI) * Gm[parentcomp]:t() + torch.diag(PsiI:pow(-1))
   local delta0 = distributions.mvn.rnd(torch.zeros(1, p), cov)
   local delta = self.GmpX_star_N[parentcomp] + delta0:view(p, 1):expand(p, N)
   local assign = torch.sign(torch.cmul(delta, self.PtMuDiff_N[parentcomp]):sum(1))

   local Qss = torch.zeros(N)
   Qss[assign:eq(1)] = Qsp[assign:eq(1)] * self.hardness
   Qsp[assign:eq(1)] = Qsp[assign:eq(1)] * (1 - self.hardness)
   Qss[assign:eq(-1)] = Qsp[assign:eq(-1)] * (1 - self.hardness)
   Qsp[assign:eq(-1)] = Qsp[assign:eq(-1)] * self.hardness
   self.Qs_N = self.Qs_N:cat(Qss)

   parent:addComponent(parentcomp)
end


function CMFA:handleDeath()
   local n, S, p, k, d = self:_setandgetDims()
   local Qs_N = self.Qs_N
   local comp2remove = Set(torch.find(Qs_N:sum(1):lt(100), 1))
   local comp2keep = Set(torch.find(Qs_N:sum(1):lt(100), 0))
   local numDeaths = Set.len(comp2remove)

   if numDeaths == 0 then return false end

   print(string.format('Removing following components = %s\n', comp2remove))
   self:removeComponent(torch.LongTensor(Set.values(comp2keep)))
   return true
end


function CMFA:removeComponent(comp2keep)
   self.Xm_N = self.Xm_N:index(1, comp2keep)
   self.Zm_N = self.Zm_N:index(1, comp2keep)
   self.GmpX_star_N = self.GmpX_star_N:index(1, comp2keep)
   self.PtMuDiff_N = self.PtMuDiff_N:index(1, comp2keep)
   self.Qs_N = self.Qs_N:index(2, comp2keep)

   parent:removeComponent(comp2keep)
end


function CMFA:keepBirth(Y, X_star)
   local Mu = Y.Mu
   local Pt = Y.Pt

   local F, dF = self:calcF(Mu, Pt, X_star)

   if F > self.F_old then
      self.F_old = F
      return true
   else
      return false
   end
end


function CMFA:converged(target, delta)
   if delta == math.huge or torch.abs(delta / target) > 0.05 then
      return false
   else
      return true
   end
end

------------------------------------------------------
-- function to handle birth-death heuristic
-- It should be called after all the batches have
-- finished
------------------------------------------------------
function CMFA:handleBirthDeath()
   local F, dF = self:calcF()
   print(string.format('F = %f\tdF = %f\n', F, dF))

   if self:converged(F, dF) then
      if not self.inbirthprocess then
         print('No birth convergence. Trying birth-death.')
         local orderanddobirth = true

         if not self.lastbirthsuccess and self.candidx <= self.order:size(2) - 1 then
            self.candidx = self.candidx + 1
            local newcand = self.order[1][self.candidx]
            print(string.format('Last birth failed. Trying birth with new candidate = %d', newcand))
            self.Fold = F
            self:handleBirth(newcand)
            self.inbirthprocess = true
            orderanddobirth = false
         else
            print('Trying death')
            self.deathstatus = self:handleDeath()
            orderanddobirth = not self.deathstatus
            print(string.format('Death status = %s', self.deathstatus))
         end

         if orderanddobirth then
            self.order = self:orderCandidates()
            print(string.format('New order of birth candidates = \n%s', self.order))
            self.candidx = 1
            print(string.format('Reordering and trying birth with candidate = %d', self.order[1][self.candidx]))
            self.Fold = F
            self:handleBirth(self.order[1][self.candidx])
            self.inbirthprocess = true
         end

         return false
      else
         print('Birth convergence. Checking for lower bound')
         if F > self.Fold then
            print('Keeping birth')
            self.Fold = F
            self.lastbirthsuccess = true
            self.inbirthprocess = false
            return false -- ask nn to keep things
         else
            print('Reverting previous birth')
            self.lastbirthsuccess = false
            self:loadWorkspace('cfmaworkspace.dat') -- revert cmfa to prev state
            self.inbirthprocess = false
            return true -- ask nn to revert things
         end
      end
      print(string.format('\n'))
   else
      print(string.format('Not converged\n'))
      if self.inbirthprocess then
         if self.Qs_N:sum(1):lt(10):sum() > 0 then
            print('Responsibilities are below threshold. Reverting previous birth.')
            self:loadWorkspace('cfmaworkspace.dat')
            self.lastbirthsuccess = false
            self.inbirthprocess = false
         end
      end
      return false
   end
end



return CMFA