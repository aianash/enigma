require 'torch'

enigma = {}

-- Include global files
torch.include('enigma', 'task.lua')

-- Include each tasks
torch.include('enigma', 'feature/Feature.lua')

return enigma