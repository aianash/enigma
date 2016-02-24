require 'torch'

local Feature, parent = torch.class('enigma.Feature', 'enigma.Task')

function Feature:__init(opt)
	parent.__init(self, "Feature")
end

function Feature:begin()
	print("Beginning Feature task")
end