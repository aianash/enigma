------------------
--[[ Identity ]]--
--
------------------
local Identity = klazz("enigma.mobius.Identity")
Identity.isIdentity = true

function Identity:forward(inputs) return inputs end
function Identity:backward() end
