require 'torch'

local enigma = require 'enigma'
local lapp = require 'pl.lapp'

torch.setdefaulttensortype('torch.FloatTensor')

local opt = lapp [[
Enigma training script for core AI model
Main options
	--feature 						Run feature training task
	--search 						Run search training task
	--preprocessing 				Dataset preprocessing task
	-o, --output 					Output file for 
	-d, --dataset 					Name of the dataset to use
	-e, --epochs 					Number of epochs to run

Feature Training options
	--glimpses 						Comma seperated glimpses config with their weights

Search Training options
	--glimpses 						Comma seperated glimpses config with their weights

Preprocessing Task options

]]

local task

if opt.feature then
	task = enigma.Feature(opt)
elseif opt.complete then
	task = enigma.Search(opt)
elseif opt.preprocessing then
	task = enigma.Preprocessing(opt)
end

task:print()
task:begin()