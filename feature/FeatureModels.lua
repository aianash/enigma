require 'pl'
local cjson = require 'cjson'

-- Include abstract model class and its implementations
torch.include('enigma', 'feature/FeatureModel.lua')
torch.include('enigma', 'feature/FeatureGlimpseModel.lua')

---------------------------------------------------
--[[ enigma.FeatureModels ]]--
-- Model Factory class
---------------------------------------------------
local FeatureModels = torch.class('enigma.feature.FeatureModels')
FeatureModels.isFeatureModels = true

function FeatureModels:get(name, cmdOpt)
	local configPath = path.join(path.currentdir(), cmdOpt.configDir, "feature_"..name..".json")
	if not path.isfile(configPath) then
		error(string.format('No model config at %s', configPath))
	end

	local config = cjson.decode(file.read(configPath))
	local model
	if name == 'feature-glimpse' then
		model = enigma.feature.FeatureGlimpseModel(config, cmdOpt)
	end
	return model
end