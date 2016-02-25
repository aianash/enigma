require 'torch'

enigma = {
	feature = {}, -- feature
	dataset = {} -- dataset
}

-- Include global files
torch.include('enigma', 'ItemTypes.lua')
torch.include('enigma', 'Task.lua')

-- Include datasets
torch.include('enigma', 'dataset/Datasets.lua')

-- Include feature
torch.include('enigma', 'feature/Feature.lua')



return enigma