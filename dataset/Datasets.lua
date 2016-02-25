local pl = (require 'pl.import_into')()
pl.stringx.import()

local d = import('enigma.dataset')

-----------------------------------------------
--[[ enigma.Datasets ]]--
-- Factory class to get named dataset
-----------------------------------------------
local Datasets = klazz('enigma.dataset.Datasets')
Datasets.isDatasets = true 

--
function Datasets:get(name, source, argstr)
	local dataset
	local args = {}
	if type(argstr) == 'string' then
		args = argstr:split(',')
	end

	if name == d.RawItemImageIntentDataset.name then
		dataset = d.RawItemImageIntentDataset(source, args[1], args[2])
	elseif name == d.ImageGlimpseIntentVectorDataset.name then
		dataset = d.ImageGlimpseIntentVectorDataset(source, args[1])
	end

	return dataset
end

--
function Datasets:createNew(name)
	-- body
end

return Datasets