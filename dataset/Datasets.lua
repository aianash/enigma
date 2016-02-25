local pl = (require 'pl.import_into')()

pl.stringx.import()

torch.include('enigma', 'dataset/Dataset.lua')
torch.include('enigma', 'dataset/RawItemImageIntentDataset.lua')
torch.include('enigma', 'dataset/ImageGlimpseIntentVectorDataset.lua')

-----------------------------------------------
--[[ enigma.Datasets ]]--
-- Factory class to get named dataset
-----------------------------------------------
local Datasets = torch.class('enigma.dataset.Datasets')
Datasets.isDatasets = true 

--
function Datasets:get(name, source, argstr)
	local dataset
	local args = {}
	if type(argstr) == 'string' then
		args = argstr:split(',')
	end

	if name == enigma.dataset.RawItemImageIntentDataset.name then
		dataset = enigma.dataset.RawItemImageIntentDataset(source, args[1], args[2])
	elseif name == enigma.dataset.ImageGlimpseIntentVectorDataset.name then
		dataset = enigma.dataset.ImageGlimpseIntentVectorDataset(source, args[1])
	end

	return dataset
end

--
function Datasets:createNew(name)
	-- body
end