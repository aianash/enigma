----------------------------------------------------------
-- conditional mixture of factor analyser
----------------------------------------------------------
require 'gnuplot'

local distributions = require 'distributions'
local display = require 'display'
local VBCMFA = require 'VBCMFA'

-- configure display
display.configure({hostname='127.0.0.1', port=8000})

local CMFA = {}

local parent = VBCMFA:_factory()
setmetatable(CMFA, parent)
parent.__index = parent


function CMFA:new(...)
   local o = {}
   setmetatable(o, self)
   self.__index = self
   o:__init(...)
   return o
end


function CMFA:__init(cfg)
   self.debug = cfg.debug
   self.pause = cfg.pause
   parent:__init(cfg)
end


-- Mu  means of factor analyzers of intent
-- Pti responsibilities
function CMFA:train(Mu, Pt, X_star, epochs)
   local n, s, p, k, d, N = self:_setandgetDims()
   self.old = {}
   self.Fhist = torch.zeros(epochs, 1)

   self.conditional.Xm:copy(X_star:repeatTensor(s, 1, 1))

   local order = torch.Tensor(0)
   local orderingRequired = true
   local birthStatus = true
   local cand = 1

   for epoch = 1, epochs do
      self:learn(Mu, Pt, X_star, epoch, 20, 5)

      local F, dF = self:calcF(self.debug, Mu, Pt, X_star)
      print(string.format("F = %f\tdF = %f\n", F, dF))
      self.Fhist[epoch] = F

      if self:converged(F, dF) then
         print("Converged, Trying birth-death")
         local deathStatus = false

         if orderingRequired then
            order = self:orderCandidates()
            print(string.format("Order of candidates = \n%s\n", order))
            orderingRequired = false
         end

         if not birthStatus and cand <= order:size(2) then
            print(string.format("Trying birth with candidate number = %d\tParent = %d\n", cand, order[1][cand]))
            birthStatus = self:handleBirth(Mu, Pt, X_star, order[1][cand])
            print(string.format("Birth status = %s\n", tostring(birthStatus)))
         else
            print(string.format("Trying removal."))
            deathStatus = self:handleRemoval()
            print(string.format("Death status = %s\n", tostring(deathStatus)))
         end

         if not deathStatus then
            print("Reordering candidates for birth")
            order = self:orderCandidates()
            cand = 1
            print(string.format("New order of candidates = \n%s\n", order))
            print(string.format("Trying birth with candidate number = %d\tParent = %d\n", cand, order[1][cand]))
            birthStatus = self:handleBirth(Mu, Pt, X_star, order[1][cand])
            print(string.format("Birth status = %s\n", tostring(birthStatus)))
            if not birthStatus then
               cand = cand + 1
            end
         end
      end

      collectgarbage()
   end

   self:plotFhist()
   return self.F, self.factorLoading.Gm, self.factorLoading.Gcov, self.factorLoading.Lm, self.factorLoading.Lcov, self.hidden.Zm, self.conditional.Xm, self.hidden.Qs
end


function CMFA:converged(target, delta)
   -- if epoch % 10 == 0 then return true else return false end
   if delta == math.huge then
      return false
   elseif torch.abs(delta / target) < 0.005 then
      return true
   else
      return false
   end
end


function CMFA:handleBirth(Mu, Pt, X_star, parent)
   local file = 'workspace.dat'
   local Ftarget = self:calcF(self.debug, Mu, Pt, X_star)
   self:saveWorkspace(file)
   self:addComponent(parent, Mu, Pt)
   local i = 1
   while true do
      self:learn(Mu, Pt, X_star, i, 20, 5)
      i = i + 1
      local F, dF = self:calcF(self.debug, Mu, Pt, X_star)
      if self:converged(F, dF) then break end
   end
   local F = self:calcF(self.debug, Mu, Pt, X_star)
   if F > Ftarget then
      print(string.format("Keeping the birth\n"))
      return true
   else  -- revert to previous state
      print(string.format("Reverting to previous state\n"))
      self:loadWorkspace(file)
      return false
   end
end


function CMFA:learn(Mu, Pt, X_star, epoch, convEpoch, subEpoch)
   for convEpoch = 1, 20 do
      self:inferQXZ(Mu, Pt, X_star, 10)
      self:inferQLG(Mu, Pt, 10)

      for subEpoch = 1, 10 do
         self:infer("Qnu")
         self:infer("Qomega")
      end

      self:infer("Qs", Mu, Pt)
      self:infer("Qpi")

      if epoch % 2 == 0 then
         for subEpoch = 1, 5 do
            self:infer("PsiI", Mu, Pt)
            self:infer("HyperX", X_star)
         end
      end
   end

   self:infer("Qs", Mu, Pt)
end


function CMFA:plotFhist()
   local config = {
      title = "Lower Bound (F)",
      lables = {"Epochs", "F"}
   }
   local Fsize = self.Fhist:size(1)
   local data = torch.cat(torch.linspace(1, Fsize, Fsize), self.Fhist, 2)
   display.plot(data, config)
end


function CMFA:infer(tr, ...)
   self["infer"..tr](self, ...)
end


function CMFA:print(tr, ns)
   print(string.format("%s after inference = \n", tr))
   print(self[ns][tr])
   if self.pause == 1 then io.read() end
end


return CMFA