local Datasets = enigma.dataset.Datasets()

-------------------------------------------------------------------------
--[[ enigma.FeatureGlimpseModel ]]--
-- Deep Neural Network model identify and normalize features
-- This model uses Spatial Transformer Network with complete Affine
-- projection
-------------------------------------------------------------------------
local FeatureGlimpseModel, parent = torch.class('enigma.feature.FeatureGlimpseModel', 'enigma.feature.FeatureModel')
FeatureGlimpseModel.isFeatureGlimpseModel = true

function FeatureGlimpseModel:__init(config, cmdOpt)
	parent:__init("Feature Glimpse", "Description", config)
	self.epochs = self.config.epochs

	self.dataset = Datasets:get(cmdOpt.dataset, cmdOpt.datasetSource, cmdOpt.datasetArgs)
	if not self.dataset or not self.dataset.isImageGlimpseIntentVectorDataset then
		error('This model requires '..enigma.dataset.ImageGlimpseIntentVectorDataset.name..' dataset')
	end
end

function FeatureGlimpseModel:train()
	print("Training will start here")
	print(self.config.glimpses[1].x)
	print(self.epochs)
end

function FeatureGlimpseModel:test()
	-- body
end