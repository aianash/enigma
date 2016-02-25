require 'torch'

local enigma = require 'enigma'
local lapp = require 'pl.lapp'

torch.setdefaulttensortype('torch.FloatTensor')

local opt = lapp [[
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

local task

if opt.feature then
	task = enigma.feature.Feature(opt)
elseif opt.complete then
	task = enigma.Search(opt)
elseif opt.preprocessing then
	task = enigma.Preprocessing(opt)
end

task:print()
task:begin()