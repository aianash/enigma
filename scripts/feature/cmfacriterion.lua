local CMFACriterion, parent = torch.class('nn.CMFACriterion', 'nn.Criterion')

--
function CMFACriterion:__init(cmfa)
   parent.__init(self)
   self.cmfa = cmfa
end

--
function CMFACriterion:forward(X_star, Y, batchperm, batchIdx, epochs) -- n x f, n x p
   local likelihood = self.cmfa:batchtrain(Y:t(), X_star:t():contiguous(), batchperm, batchIdx, epochs)
   self.output = - likelihood
   return self.output
end

--
function CMFACriterion:backward(X_star, Y) -- n x f, n x p
   local cmfa = self.cmfa
   local Xm = cmfa.conditional:getX() -- s x f x n
   local Qs = cmfa.hidden:getS() -- n x s
   local E_starI = cmfa.hyper.E_starI

   local s = cmfa.s
   local N, f = X_star:size(1), X_star:size(2)

   local gradInput = torch.zeros(N, f)

   local Xm_X_star = torch.bmm(Xm:permute(3, 2, 1), Qs:view(N, s, 1)):squeeze() - X_star  -- n x f
   self.gradInput = Xm_X_star * E_starI -- n x f
   return self.gradInput
end

--
function CMFACriterion:reset()
   self.cmfa:reset()
end

return CMFACriterion