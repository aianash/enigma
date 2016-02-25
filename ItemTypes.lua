-------------------------------------------------
--[[ enigma.ItemTypes ]]--
-- Defines all item types recognized by enigma
-- Every task is run independently for a given item type
-------------------------------------------------
local ItemTypes = torch.class('enigma.ItemTypes')
ItemTypes.isItemTypes = true

-- matches/validates and returns identifier for
-- the given command string for item type
-- command string is a comman seperated values
-- to locate item type
function ItemTypes:get(cmdStr)

end