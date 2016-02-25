---------------------------------------------
--[[ enigma.Dataset ]]--
-- Abstract class for all datasets
---------------------------------------------
local Dataset = torch.class('enigma.dataset.Dataset')
Dataset.isDataset = true

--
function Dataset:__init()
	
end

--
function Dataset:complete()
	error('Not implemented')
end

--
function Dataset:training()
	error('Not implemented')
end

--
function Dataset:test()
	error('Not implemented')
end

--
function Dataset:validation()
	error('Not implemented')
end