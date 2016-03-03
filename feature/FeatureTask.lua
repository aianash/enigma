---------------------------------------------------------------------
--[[ enigma.Feature ]]--
-- This class performs feature related tasks that include
-- 1. Learning to identify and normalize (spatial transformation) individual
--      features provided glimpses. These are used as pre-trained models
--    for complete search training.
---------------------------------------------------------------------
local FeatureTask, parent = klazz('enigma.feature.FeatureTask', 'enigma.Task')
FeatureTask.isFeatureTask = true

-- singleton instance
local FeatureModels = enigma.feature.FeatureModels()

-- Various opts (command line) pre-configures the task parameters
-- This include
--[[
Main options
--train  (string)             Name of the model to train
--configDir (default "./config")    Path to directoy containing model config files (details below)
]]--
function FeatureTask:__init(cmdOpt)
   if type(cmdOpt.train) ~= 'string' then
      error('Must provide the name of the model to train')
   end
   
   self.model = FeatureModels:get(cmdOpt.train, cmdOpt)
   parent.__init(self, "FeatureTask", self:mkDescription(), cmdOpt)
end

function FeatureTask:begin()
   print("Beginning Feature Task")
   self.model:train()
end

function FeatureTask:mkDescription()
   return string.format("Feature task running for model [%s]\n%s\n", self.model.name, self.model.description)
end

return FeatureTask