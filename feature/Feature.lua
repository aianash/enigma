---------------------------------------------------------------------
--[[ enigma.Feature ]]--
-- This class performs feature related tasks that include
-- 1. Learning to identify and normalize (spatial transformation) individual
-- 	  features provided glimpses. These are used as pre-trained models
--    for complete search training.
---------------------------------------------------------------------
local Feature, parent = klazz('enigma.feature.Feature', 'enigma.Task')
Feature.isFeature = true

-- singleton instance
local FeatureModels = enigma.feature.FeatureModels()

-- Various opts (command line) pre-configures the task parameters
-- This include
--[[
Main options
--train 	(string)					Name of the model to train
--configDir (default "./config")		Path to directoy containing model config files (details below)
]]--
function Feature:__init(cmdOpt)
	if type(cmdOpt.train) ~= 'string' then
		error('Must provide the name of the model to train')
	end
	
	self.model = FeatureModels:get(cmdOpt.train, cmdOpt)
 	parent.__init(self, "Feature", self:mkDescription(), cmdOpt)
end

function Feature:begin()
	print("Beginning Feature Task")
	self.model:train()
end

function Feature:mkDescription()
	return string.format("Feature task running for model [%s]\n%s\n", self.model.name, self.model.description)
end

return Feature