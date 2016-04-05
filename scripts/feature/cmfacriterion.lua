local CMFACriterion, parent = torch.class('nn.CMFACriterion', 'nn.Criterion')

--
function CMFACriterion:__init(cmfa)
   parent.__init(self)
   self.cmfa = cmfa
end

--
function CMFACriterion:updateOutput(input, target) -- n x f, n x p
   local likelihood = self.cmfa:train(target:t(), input:t(), 3)
   self.output = - likelihood
   return self.output
end

--
function CMFACriterion:updateGradInput(input, target) -- n x f, n x p
   local cmfa = self.cmfa
   local Xm = cmfa.conditional:getX() -- s x f x n
   local Qs = cmfa.hidden:getS() -- n x s
   local E_starI = cmfa.hyper.E_starI

   local s = cmfa.s
   local N, f = input:size(1), input:size(2)

   local gradInput = torch.zeros(N, f)

   local Xm_X_star = torch.bmm(Xm:permute(3, 2, 1), Qs:view(n, s, 1)):squeeze() - input  -- n x f
   self.gradInput = Xm_X_star * E_starI -- n x f
   return self.gradInput
end

return CMFACriterion