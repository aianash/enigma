local pl = (require 'pl.import_into')()
pl.stringx.import()

-- get individual dataset and pack them to
-- create a map table
local datasets = table.pack(import  [[
                                       enigma.dataset.{
                                          RawItemImageIntentDataset,
                                          ImageGlimpseIntentVectorDataset
                                       }
                                    ]])

-----------------------------------------------
--[[ enigma.Datasets ]]--
-- Factory class to get named dataset
-----------------------------------------------
local Datasets = klazz('enigma.dataset.Datasets')
Datasets.isDatasets = true 

Datasets._map = {}
for _, dataset in ipairs(datasets) do
   Datasets._map[dataset.name] = dataset
end

--
function Datasets:get(name, source, argstr)
   local args = {}
   if type(argstr) == 'string' then args = argstr:split(',') end
   return Datasets._map[name](source, args)
end

--
function Datasets:createNew(name)
   -- body
end

return Datasets