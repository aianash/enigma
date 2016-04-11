local pl = require('pl.import_into')()

-- from fblualib/fb/util/data.lua , copied here because fblualib is not rockspec ready yet.
-- deepcopy routine that assumes the presence of a 'clone' method in user
-- data should be used to deeply copy. This matches the behavior of Torch
-- tensors.
local function deepcopy(x)
    local typename = type(x)
    if typename == "userdata" then
        return x:clone()
    end
    if typename == "table" then
        local retval = { }
        for k,v in pairs(x) do
            retval[deepcopy(k)] = deepcopy(v)
        end
        return retval
    end
    return x
end

-----------------
--[[ Mobius ]]--
-----------------
local mobius = klazz("enigma.mobius.Mobius")

----------------------------------------------------------------------------------
--[[ MobiusJunction ]]--
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
local MobiusJunction = klazz("enigma.mobius.MobiusJunction")

--
function MobiusJunction:__init(cfg)
   -- Assumed only one level of nesting
   self.branches = {}
   for idx, branch in ipairs(cfg.topology) do
      self.branches[idx] = branch.parent
   end

   self.iterCounter = self.iterationCounter(cfg.iterations)
   self.batchSize = cfg.batchSize
end

--
function MobiusJunction:__mobiuschain(inputs, final)
   -- Forward all
   local outputs = {}
   for idx, branch in ipairs(self.models) do
      outputs[idx] = branch.model:forward(inputs[idx])
   end

   -- Backward and optimize one by one
   for idx, branch in ipairs(self.models) do
      local err = branch.model:backward(inputs, outputs, false) -- [TO FIX] the branch models are assumed to know details of outputs and no grad parameter
      if branch.optimizer then branch.optimizer:optimize(err) end
      outputs[idx] = branch.model:forward(inputs[idx])
   end

   -- final backward results in err and gradients
   if final then
      local errors = {}
      local grads = {}
      for idx, branch in ipairs(self.models) do
         local err, grad = branch.model:backward(inputs, outputs, true)
         errors[idx] = err
         grads[idx] = grad
      end
      return errors, grads
   end
end

-- Train the models in mobius junction
-- here inputs is considered as data point
-- returns
-- gradient  with respect to inputs of each module
-- errors    for each module
function MobiusJunction:train(inputs)
   -- run mobius chain iteratively
   local numIter = self.iterCounter:nextNumIter() - 1
   for iter = 1, numIter do
      self:__mobiuschain(inputs)
   end
   return self:__mobiuschain(inputs, true) -- final execution
end

function MobiusJunction.iterationCounter(cfg)
   return { -- [TO DO] always returning constant
      nextNumIter = function() return 10 end,
      reset = function() end
   }
end

function MobiusJunction:resetIterationCounter()
   self.iterCounter:reset()
end