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
local mobius = {}

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
local MobiusJunction = {}
function MobiusJunction:new(...)
   local o = {}
   setmetatable(o, self)
   self.__index = self
   o:__init(...)
   return o
return

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
local MobiusTrainer = {}
function MobiusTrainer:new(...)
   local o = {}
   setmetatable(o, self)
   self.__index = self
   o:__init(...)
   return o
end

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


-----------------
--[[ Nothing ]]--
-- 
-----------------
local Nothing = { isNothing = true }
function Nothing:new()
   local o = {}
   setmetatable(o, self)
   self.__index = self
   return o
end

function Nothing:forward()
   -- body
end

function Nothing:backward()
   -- body
end


------------------
--[[ Identity ]]--
--
------------------
local Identity = { isIdentity = true }
function Identity:new()
   local o = {}
   setmetatable(o, self)
   self.__index = self
   return o
end

function Identity:forward(inputs)
   return inputs
end

function Identity:backward()
   -- body  
end

------------------------
--[[ NNOptim ]]--
-- Optimizer for Neural Network models
-- This implementation is a modification
-- over fbnn's nn.Optim, for the mobius learning
-- method
------------------------
local NNOptim = {}
do -- creating local scope for local variables
   function NNOptim:new(...)
      local o = {}
      setmetatable(0, self)
      self.__index = self
      o:__init(...)
      return o
   end

   -- Returns weight parameters and bias parameters and associated grad parameters
   -- for this module. Annotates the return values with flag marking parameter set
   -- as bias parameters set
   function NNOptim.weight_bias_parameters(module)
       local weight_params, bias_params
       if module.weight then
           weight_params = {module.weight, module.gradWeight}
           weight_params.is_bias = false
       end
       if module.bias then
           bias_params = {module.bias, module.gradBias}
           bias_params.is_bias = true
       end
       return {weight_params, bias_params}
   end

   -- [TO DO] no checkpoint supported yet
   function NNOptim:__init(model, optimMethod, optState)
      
      self.model = model
      self.optimMethod = optimMethod

      self.originalOptState = optState

      self.modulesToOptState = {}
      self.model:for_each(function(module)
         self.modulesToOptState[module] = {}
         local params = self.weight_bias_parameters()
         -- expects either an empty table or 2 element table, one for weights
         -- and one for biases
         assert(pl.tablex.size(params) == 0 or pl.tablex.size(params) == 2)
         for i, _ in ipairs(params) do
             self.modulesToOptState[module][i] = deepcopy(optState)
             if params[i] and params[i].is_bias then
                 -- never regularize biases
                 self.modulesToOptState[module][i].weightDecay = 0.0
             end
         end
         assert(module)
         assert(self.modulesToOptState[module])
      end)
   end

   local function get_device_for_module(mod)
      local dev_id = nil
      for name, val in pairs(mod) do
          if torch.typename(val) == 'torch.CudaTensor' then
              local this_dev = val:getDevice()
              if this_dev ~= 0 then
                  -- _make sure the tensors are allocated consistently
                  assert(dev_id == nil or dev_id == this_dev)
                  dev_id = this_dev
              end
          end
      end
      return dev_id -- _may still be zero if none are allocated.
   end

   local function on_device_for_module(mod, f)
      local this_dev = get_device_for_module(mod)
      if this_dev ~= nil then
         return cutorch.withDevice(this_dev, f)
      end
      return f()
   end

   --
   function NNOptim:optimize(err)
      local curGrad
      local curParam
      local function fEvalMod(x)
         return err, curGrad
      end 

      for curMod, opt in pairs(self.modulesToOptState) do
         on_device_for_module(curMod, function()
            local curModParams = self.weight_bias_parameters(curMod)

            assert(pl.tablex.size(curModParams) == 0 or
                  pl.tablex.size(curModParams) == 2)
            if curModParams then
               for i, tensor in ipairs(curModParams) do
                  if curModParams[i] then
                     -- expect param, gradParam pair
                     curParam, curGrad = table.unpack(curModParams[i])
                     assert(curParam and curGrad)
                     self.optimMethod(fEvalMod, curParam, opt[i])
                  end
               end
            end
         end)
      end
   end
end

-----------------------------
--[[ MFAOptim ]]--
-- This optimizer updates the hyperparameters
-- of the VBMFA model, after posteriors have
-- been inferred in backward step, by the model
-----------------------------
local MFAOptim = {}
do -- starting local context for various helper functions
   function MFAOptim:new(...)
      local o = {}
      setmetatable(o, self)
      self.__index = self
      o:__init(...)
      return o
   end

   --
   function MFAOptim:__init(model)
      
   end

   --
   function MFAOptim:optimize(err)

   end

end

mobius.Nothing = Nothing
mobius.Identity = Identity
mobius.MobiusTrainer = MobiusTrainer
mobius.NNOptim = NNOptim
mobius.MFAOptim = MFAOptim

return mobius