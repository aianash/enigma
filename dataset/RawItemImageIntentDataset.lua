local pl = (require 'pl.import_into')()
require 'image'

---------------------------------------------------------------------------
--[[ enigma.dataset.RawItemImageIntentDataset ]]--
-- Source for this dataset is a directory/zip containing
-- individual image's file and a csv containing labels
-- csv columns are
-- [Filename, Relevant Intent, Irrelevant Intent]
--------------------------------------------------------------------------- 
local RawItemImageIntentDataset, parent = klazz('enigma.dataset.RawItemImageIntentDataset', 'enigma.dataset.Dataset')
RawItemImageIntentDataset.isRawItemImageIntentDataset = true
RawItemImageIntentDataset.name = 'raw-item-image-intent'

-- path to either directory or zip containg the data
-- filename of the csv
-- root directory name if zip file was provided as source
function RawItemImageIntentDataset:__init(src, args)
   local labelcsv = args[1];
   local rootdirectoryName = args[2]
   local source = src
   if pl.path.extension(source) == 'zip' and rootdirectoryName then
      os.execute('unzip '..source..';')
      source = pl.path.join(pl.path.currentdir, rootdirectoryName) 
   elseif not pl.path.isdir(source) then
      error('Source path should either be a zip file with a root dir or a directory')
   end

   self._labelcsvContent = pl.data.read(pl.path.join(source, labelcsv))
   self.source = source
   self.size = #self._labelcsvContent
end

-- returns an iterator with elements
-- (imageData, relevant intent, irrelevant intent)
function RawItemImageIntentDataset:complete()
   local filenameIdx = self._labelcsvContent.fieldnames:index('Filename')
   local relIntentIdx = self._labelcsvContent.fieldnames:index('Relevant Intent')
   local irrelIntentIdx = self._labelcsvContent.fieldnames:index('Irrelevant Intent')

   local csvEntries = ipairs(self._labelcsvContent)

   return function() 
      for _, metadata in csvEntries do
         local imagePath = pl.path.join(self.source, metadata[filenameIdx])
         if not pl.path.isfile(imagePath) then
            print('[WARNING] Image not found at path '..imagePath)
         else
            local imageData = image.load(imagePath)
            return imageData, metadata[relIntentIdx], metadata[irrelIntentIdx]
         end
      end
      return nil
   end
end

-- 
function RawItemImageIntentDataset:training(num)
   error('No training dataset, use complete')
end

--
function RawItemImageIntentDataset:test(num)
   error('No test dataset, use complete')
end

--
function RawItemImageIntentDataset:validation(num)
   error('No validation dataset, use complete')
end

return RawItemImageIntentDataset