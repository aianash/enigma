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

print [[
---------------------------------
           E N I G M A
---------------------------------           
        _,    _   _    ,_
   .o888P     Y8o8Y     Y888o.
  d88888      88888      88888b
 d888888b_  _d88888b_  _d888888b
 8888888888888888888888888888888
 8888888888888888888888888888888
 YJGS8P"Y888P"Y888P"Y888P"Y8888P
  Y888   '8'   Y8P   '8'   888Y
   '8o          V          o8'
     `                     `
]]

local task

if opt.feature then
	task = enigma.feature.FeatureTask(opt)
end

-- task:print()
task:begin()