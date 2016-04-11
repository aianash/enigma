local pl = (require 'pl.import_into')()
require 'image'
require 'imgraph'

--
local Glimpser = {}

--
function Glimpser:new( ... )
   local o = {}
   setmetatable(o, self)
   self.__index = self
   o:__init(...)
   return o
end

--
function Glimpser:__init( ... ) end

--
function Glimpser:reduce(inputimg, epoch, gaussian)
   local epoch = epoch or 1
   local kernel = gaussian or image.gaussian(9)
   local inputimggrey = image.rgb2y(inputimg):repeatTensor(3, 1, 1)
   local mstsegm
   for itr = 1, epoch do
      local inputimgg = image.convolve(inputimggrey, kernel, 'same')
      local graph = imgraph.graph(inputimgg, 4)
      mstsegm = imgraph.segmentmst(graph, 3, 10)
      inputimggrey = imgraph.histpooling(inputimggrey, mstsegm)
   end

   return imgraph.histpooling(inputimg, mstsegm)
end

-----------------
--[[ Testing ]]--
-----------------
display = require 'display'

local testDir = "./glimpses-test-images"

local epoch = 1
local files = pl.dir.getfiles(testDir)
local winI
local glimpser = Glimpser:new()

for idx, imagefile in ipairs(files) do
   local inputimg = image.load(imagefile)
   inputimg = image.scale(inputimg, inputimg:size(3) / 2, inputimg:size(2) / 2)
   local reduced = glimpser:reduce(inputimg)
   display.images({inputimg, reduced})
end

return Glimpser