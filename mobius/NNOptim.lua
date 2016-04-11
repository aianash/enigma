
------------------------
--[[ NNOptim ]]--
-- Optimizer for Neural Network models
-- This implementation is a modification
-- over fbnn's nn.Optim, for the mobius learning
-- method
------------------------
local NNOptim = klazz("enigma.mobius.NNOptim")

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