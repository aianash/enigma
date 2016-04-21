local pl = require('pl.import_into')()
require 'nn'
require 'optim'
local VBCMFACriterion = enigma.cmfa.VBCMFACriterion

--------------------------------------------------
--[[ Neural Network VBCMFA Trainer ]]--
-- Trains a neural network with CMFA criterion
--------------------------------------------------
local NNCMFATrainer = klazz("enigma.cmfa.NNVBCMFATrainer")

--
function NNCMFATrainer:__init(network, cmfa, optState)
   self.network = network
   self.cmfa = cmfa
   self.criterion = VBCMFACriterion(cmfa)

   print(self.network)
   self.originalOptState = optState

   self.modulesToOptState = {}
   self.network:apply(function(module)
      self.modulesToOptState[module] = {}
      local params = self.weight_bias_parameters(module)
      -- expects either an empty table or 2 element table, one for weights
      -- and one for biases
      assert(pl.tablex.size(params) == 0 or pl.tablex.size(params) == 2)
      for i, _ in ipairs(params) do
          self.modulesToOptState[module][i] = Utils.deepcopy(optState)
          if params[i] and params[i].is_bias then
              -- never regularize biases
              self.modulesToOptState[module][i].weightDecay = 0.0
          end
      end
      assert(module)
      assert(self.modulesToOptState[module])
   end)
end

-- Returns weight parameters and bias parameters and associated grad parameters
-- for this module. Annotates the return values with flag marking parameter set
-- as bias parameters set
function NNCMFATrainer.weight_bias_parameters(module)
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

--
function NNCMFATrainer:train(L, D, epochs, cmfaepochs) -- N x p, -- N x 1 x h x w
   local n, s, p, k, f, N = self:sizes()

   local Lbatch = torch.zeros(n, p)
   local Dbatch = torch.zeros(n, D:size(2), D:size(3), D:size(4))

   for epoch = 1, epochs do
      print(string.format([[
--------------------------
Training Epoch %d
--------------------------]], epoch))

      local permutation = torch.linspace(1, N, N):long()--torch.randperm(N):long()
      local batches = permutation:split(n)

      self.criterion:reset()

      for batchIdx, batch in ipairs(batches) do
         print(string.format([[
Training batch %d
-----------------]], batchIdx))

         -- prepare batches
         Lbatch:index(L, 1, batch)
         Dbatch:index(D, 1, batch)

         self:optimize(optim.sgd, Dbatch, Lbatch, batch, batchIdx, cmfaepochs)
      end
   end
end

--
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

--
local function on_device_for_module(mod, f)
   local this_dev = get_device_for_module(mod)
   if this_dev ~= nil then
      return cutorch.withDevice(this_dev, f)
   end
   return f()
end

--
function NNCMFATrainer:optimize(optimMethod, Dbatch, Lbatch, batchperm, batchIdx, cmfaepochs)
   self.network:zeroGradParameters()
   local X_star = self.network:forward(Dbatch) -- n x f
   local err = self.criterion:forward(X_star, Lbatch, batchperm, batchIdx, cmfaepochs)

   -- for subepoch = 1, 3 do
      local dL_dx_star = self.criterion:backward(X_star, Lbatch)
      self.network:backward(Dbatch, dL_dx_star)

      local curGrad
      local curParam
      local function fEvalMod(x)
         return err, curGrad
      end

      for curMod, opt in pairs(self.modulesToOptState) do
         on_device_for_module(curMod, function()
            local curModParams = self.weight_bias_parameters(curMod)
            -- expects either an empty table or 2 element table, one for weights
            -- and one for biases
            assert(pl.tablex.size(curModParams) == 0 or
                   pl.tablex.size(curModParams) == 2)
            if curModParams then
                for i, tensor in ipairs(curModParams) do
                    if curModParams[i] then
                        -- expect param, gradParam pair
                        curParam, curGrad = table.unpack(curModParams[i])
                        assert(curParam and curGrad)
                        optimMethod(fEvalMod, curParam, opt[i])
                    end
                end
            end
         end)
      end
   -- end
end

--
function NNCMFATrainer:sizes()
   return self.cmfa:_setandgetDims()
end

return NNCMFATrainer