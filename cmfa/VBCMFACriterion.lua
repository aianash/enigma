local VBCMFACriterion, parent = torch.class('nn.VBCMFACriterion', 'nn.Criterion')

--
function VBCMFACriterion:__init(vbcmfa)
   parent.__init(self)
   self.vbcmfa = vbcmfa
end

--
function VBCMFACriterion:forward(X_star, Y, batchperm, batchIdx, epochs) -- n x f, n x p
   local likelihood = self.vbcmfa:batchtrain(Y:t(), X_star:t():contiguous(), batchperm, batchIdx, epochs)
   self.output = - likelihood
   return self.output
end

--
function VBCMFACriterion:backward(X_star, Y) -- n x f, n x p
   local vbcmfa = self.vbcmfa
   local Xm = vbcmfa.conditional:getX() -- s x f x n
   local Qs = vbcmfa.hidden:getS() -- n x s
   local E_starI = vbcmfa.hyper.E_starI

   local S = vbcmfa.S
   local N, f = X_star:size(1), X_star:size(2)

   local gradInput = torch.zeros(N, f)

   local Xm_X_star = torch.bmm(Xm:permute(3, 2, 1), Qs:view(N, S, 1)):squeeze() - X_star  -- n x f
   self.gradInput = Xm_X_star * E_starI -- n x f
   return self.gradInput
end

--
function VBCMFACriterion:reset()
   self.vbcmfa:reset()
end

return VBCMFACriterion