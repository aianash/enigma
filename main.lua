require 'enigma'
require 'torch'
local pl = (require 'pl.import_into')()

torch.setdefaulttensortype('torch.FloatTensor')

local opt = pl.lapp [[
Run Enigma tasks for an item type (provided).
Tasks mainly include training AI models.

Main options
------------
	-i, --itemType  (string)  				Item Type for which this task is being run. (See itemTypes.lua)				
	--feature 						Run Feature task
	--search 						Run Search task
	--preprocessing 					Run Dataset preprocessing task
	-d, --dataset 	(default nil)				Name of the dataset to use
	-s, --datasetSource 	(default nil)				Path to dataset directory or file
	--datasetArgs 	(default nil)				Comma seperated args to dataset, refer individual datasets

Feature Training options
------------------------
(For detail look for comments in feature/Feature.lua)
	--train  	(string)				Name of the model to train
	--configDir	(default "./config")			Path to directory containing model config files
	-o, --output 						Output file for 


Search Training options
-----------------------

Preprocessing Task options
--------------------------

]]


print("\n\n")
local task

if opt.feature then
	task = enigma.feature.Feature(opt)
elseif opt.complete then
	task = enigma.search.Search(opt)
elseif opt.preprocessing then
	task = enigma.preprocessing.Preprocessing(opt)
end

-- task:print()
task:begin()