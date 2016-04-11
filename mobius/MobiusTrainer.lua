----------------------------------------------------------------------------------
--[[ MobiusTrainer ]]--
-- Parameters:
-- {
--    topology = {
--       [1] = {
--          model = featureSTM,
--          optimizer = mobius.NNOptim:new(featureSTM, optim.sgd, featureSTMOptState),
--          parent = {
--             model = mobius.Identity:new()
--          }
--       },
--
--       [2] = {
--          model = mobius.Nothing:new(),
--          parent = {
--             model = featureMFA,
--             optimizer = mobius.MFAOptim:new(featureMFA)
--          }
--       }
--    },
--    iterations = { name = 'exponential-backoff', max = 10, min = 5 },
--    batchSize = batchSize
-- }
--
-- [NOTE] Just one level of nesting is supported yet.
-- This second level will be where a mobius junction will reside
-- Therefore no nesting of mobius junction
----------------------------------------------------------------------------------
local MobiusTrainer = klazz("enigma.mobius.MobiusTrainer")

function MobiusTrainer:__init(cfg)
   self.topology = cfg.topology
   self.bs = cfg.batchSize
   self.mobiusJunction = MobiusJunction:new(cfg, iterCounter)
end

function MobiusTrainer:train(dataset) -- [TO DO] take dataset not just for primary

   local nbrItems = dataset.data:size(1)
   local nbrGlimpses = dataset.data:size(2)
   local gC, gH, gW = dataset.data:size(3), dataset.data:size(4), dataset.data:size(5)

   --
   local indices = torch.randperm(nbrItems):long():split(self.bs)
   -- indices[#indices] = nil

   -- holds a batch of data
   -- [IMP] glimpses are flattened here
   local batch = torch.zeros(self.bs * nbrGlimpses, gC, gH, gW, torch.getdefaulttensortype())

   -- train for each batch
   for t, ind in ipairs(indices) do
      -- create a batch
      for idx = 1, self.bs do
         batch:narrow(1, (idx - 1) * nbrGlimpses + 1, nbrGlimpses):copy(dataset.data:index(1, ind)[idx])
      end

      -- forward thru primary and secondaries' backend model
      local branchO = {}
      for idx, branch in ipairs(self.topology) do
         branch.model:zeroGradParameters()
         branchO[idx] = branch.model:forward(batch) -- there should be separate batch for each module
      end

      -- forward thru mobius jucntion, get backpropagated gradients for each modules in topology
      local errors, grads = self.mobiusJunction:train(branchO)

      -- propagate gradients thru primary and secondaries' backend model
      for idx, branch in ipairs(self.topology) do
         branch.model:backward(batch, grads[idx]) -- there should be seperate batch for each module
      end

      -- perform optim updates to each models
      -- primary
      for idx, branch in ipairs(self.topology) do
         if branch.optimizer then branch.optimizer:optimize(errors[idx]) end
      end
   end

end

-- use to reset iteration counter after
-- each epoch
function MobiusTrainer:resetIterationScheme()
   self.mobiusJunction:resetIterationCounter()
end
