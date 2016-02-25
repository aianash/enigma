local pl = (require 'pl.import_into')()

--------------------------------------------------------------------------------------
--[[ enigma.dataset.ImageGlimpseIntentVectorDataset ]]--
-- 
--------------------------------------------------------------------------------------
local ImageGlimpseIntentVectorDataset, parent = klazz('enigma.dataset.ImageGlimpseIntentVectorDataset', 'enigma.dataset.Dataset')
ImageGlimpseIntentVectorDataset.isImageGlimpseIntentVectorDataset = true
ImageGlimpseIntentVectorDataset.name = 'image-glimpse-intent-vector'

--
function ImageGlimpseIntentVectorDataset:__init(src, basefilename)
	-- assert(pl.path.isdir(src), 'Source should be a path to directory containing .bin files')
	-- assert(type(basefilename) == 'string', 'No base file name')

	-- self.trainPath = pl.path.join(src, basefilename..'_training.bin')
	-- self.testPath = pl.path.join(src, basefilename..'_test.bin')
	-- self.validationPath = pl.path.join(src, basefilename..'_validation.bin')

	-- assert(pl.path.isfile(self.trainPath), 'No training file at '..self.trainPath)
	-- assert(pl.path.isfile(self.testPath), 'No test file at '..self.testPath)

	-- if not pl.path.isfile(self.validationPath) then
	-- 	print('[WARNING] No validation data')
	-- end
end

-- returns
function ImageGlimpseIntentVectorDataset:training()
	-- return torch.load(self.trainPath)
end

--
function ImageGlimpseIntentVectorDataset:test()
	-- return torch.load(self.testPath)
end

--
function ImageGlimpseIntentVectorDataset:validation()
	-- if self.validationPath then
	-- 	return torch.load(self.validationPath)
	-- else
	-- 	return nil
	-- end
end

return ImageGlimpseIntentVectorDataset