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
end


function CMFA:batchtrain(Ybatch, X_starbatch, batchperm, batchIdx, epochs)
   local n, s, p, k, f, N = self:_setandgetDims()
   -- assert(self:_checkDimensions(Ybatch, p, n))
   -- assert(self:_checkDimensions(X_starbatch, f, n))

   print(string.format([[
'CMFA training batch %d'
]], batchIdx))

   self.conditional.Xm:copy(X_starbatch:repeatTensor(s, 1, 1))

   self:rho(batchIdx)
   print(string.format('rho = %f\n', self:rho()))

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
   -- self:computeRandomStuff(Mubatch, Ptbatch, X_starbatch, batchperm)

   self.Zm_N:indexCopy(3, batchperm, self.hidden.Zm)
   self.Qs_N:indexCopy(1, batchperm, self.hidden.Qs)

   return F
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


function CMFA:converged(taget, delta)
   if delta == math.huge or delta / target > 0.005 then
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
function CMFA:handleBirthDeath(Mu, Pt, X_star)
   local F, dF = self:calcF(Mu, Pt, X_star)

   if self:converged(F, dF) then
      if not self.inbirthprocess then
         print('No birth convergence. Trying birth-death.')
         local orderanddobirth = true

         if not self.lastbirthsuccess and self.candidx <= self.order:size(2) then
            self.candidx = self.candidx + 1
            local newcand = order[1][self.candidx]
            print(string.format('Birth with parent = %d', newcand))
            self.Fold = F
            self:handleBirth(Mu, Pt, X_star, newcand)
            self.inbirthprocess = true
            orderanddobirth = false
         else
            print('Trying death')
            self.deathstatus = self:handleDeath()
            orderanddobirth = self.deathstatus
            print(string.format('Death status = %s', self.deathstatus))
         end

         if not orderanddobirth then
            self.order = self:orderCandidates()
            self.candidx = 1
            print(string.format('Birth with parent = %d', self.candidx))
            self.Fold = F
            self:handleBirth(Mu, Pt, X_star, order[1][self.candidx])
            self.inbirthprocess = true
         end

         return false
      else
         print('Birth convergence. Checking for lower bound')
         if F > self.Fold then
            print('Keeping birth')
            self.lastbirthsuccess = true
            self.inbirthprocess = false
            return false -- ask nn to keep things
         else
            print('Reverting previous birth')
            self.lastbirthsuccess = false
            self:loadWorkspace('workspace.dat') -- revert cmfa to prev state
            self.inbirthprocess = false
            return true -- ask nn to revert things
         end
      end
   else
      return false
   end
end



return CMFA