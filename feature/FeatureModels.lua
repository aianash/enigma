local pl = (require 'pl.import_into')()
local cjson = require 'cjson'

---------------------------------------------------
--[[ enigma.FeatureModels ]]--
-- Model Factory class
---------------------------------------------------
local FeatureModels = klazz('enigma.feature.FeatureModels')
FeatureModels.isFeatureModels = true

function FeatureModels:get(name, cmdOpt)
	local configPath = pl.path.join(pl.path.currentdir(), cmdOpt.configDir, "feature_"..name..".json")
	if not pl.path.isfile(configPath) then
		error(string.format('No model config at %s', configPath))
	end

	local config = cjson.decode(pl.file.read(configPath))
	local model
	if name == 'feature-glimpse' then
		model = enigma.feature.FeatureGlimpseModel(config, cmdOpt)
	end
	return model
end

return FeatureModels