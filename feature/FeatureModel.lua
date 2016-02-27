-------------------------------------------------
--[[ enigma.feature.Model ]]--
-- Abstract class for Model (Deep Neural Network)
-------------------------------------------------
local FeatureModel = klazz('enigma.feature.FeatureModel')
FeatureModel.isFeatureModel = true

--
function FeatureModel:__init(name, description, config)
   self.name = name
   self.description = description
   self.config = config
end

--
function FeatureModel:train()
   error('Not implemented')
end

--
function FeatureModel:test()
   error('Not implemented')
end

return FeatureModel