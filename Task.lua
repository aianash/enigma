--------------------------------------
--[[ enigma.Task ]]--
-- Abstract class for all Tasks in enigma
---------------------------------------
local Task = torch.class('enigma.Task')
Task.isTask = true

local ItemTypes = enigma.ItemTypes

--
function Task:__init(name, desc, cmdOpt)
	if type(name) ~= 'string' then
		error('Task name should be provided')
	end
	self.itemType = ItemTypes:get(cmdOpt.itemType)
	self.taskName = name
	self.description = desc
end

--
function Task:print()
	print(string.format("[Task] %s", self.taskName))
	if type(self.description) == 'string' then
		print(self.description)
	end

	print("[Warning] Custom print function should be implemented by Tasks")
end

--
function Task:begin()
	error(string.format("Task:begin not implemented by task %s", self.taskName))
end