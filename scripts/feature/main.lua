local pl = (require 'pl.import_into')()
require 'image'
require 'torch'
require 'nnx'

pl.stringx.import()
torch.setdefaulttensortype('torch.FloatTensor')

-- Steps
-- A. [Optional] Take a raw dataset, explode the dataset with scaled and translated versions of images
-- B. [Optional] Iterate thru the exploded dataset and create feature glipse vectors at predefined locations
-- C. [Optional] Persist glimpse vectors in seperate bin file
-- D. [Optional] Train spatial transformer network separately for each collection of feature glimpse vectors

local explodeDataset       -- A
local createFeatureGlimpse -- B
local persisteGlipses      -- C
local trainModel           -- D


------------------------------------------------------------------------------------------------------------
--[[ A. [Optional] Take a raw dataset, explode the dataset with scale and translated versions of images ]]--
------------------------------------------------------------------------------------------------------------
do
   --[[ Config ]]--
   local srcdataset = './feature-learning-raw.zip'
   local srcRootdir = 'feature-learning-raw'
   local srccsvFileprefix = 'FLRW'

   local targetRootdir = './exploded-item-images'

   -- define Transformations on image
   local transformations = {}
   transformations[1] = function(img)
      return image.scale(img, 48, 48)
   end

   transformations[2] = function(img)
      return image.scale(image.translate(img, 5, 5), 48, 48)
   end
   --[[ End Config ]]--


   -- helper iterator for creating transformed images
   local function trimgiter(_p, transformIdx) local img = _p[1]; local filename = _p[2]
      transformIdx = transformIdx + 1
      local transform = transformations[transformIdx]
      if not transform then return nil end

      local newfilename = filename:replace(".", "_"..tostring(transformIdx):rjust(5, '0')..".")
      return transformIdx, { newfilename, transform(img) }
   end

   -- generator for transformed images
   local function tranformedimages(srcimage, filename)
      return trimgiter, { srcimage, filename }, 0 
   end

   -- This does the heavy duty of exploding dataset
   local function explode(targetRootdir, srcRootdir)
      local srcImgsubdirs = pl.dir.getdirectories(srcRootdir)
      table.sort(srcImgsubdirs)

      for _, srcImgsubdir in ipairs(srcImgsubdirs) do
         local srcImgsubdirname = pl.path.basename(srcImgsubdir)

         local srccsvFilename = srccsvFileprefix.."-"..srcImgsubdirname..'.csv'
         local srccsvContent = pl.data.read(pl.path.join(srcImgsubdir, srccsvFilename))

         -- target csv with same fields
         local targetdata = { delim = ',' }
         targetdata['fieldnames'] = srccsvContent.fieldnames

         -- create target img sub directory
         local targetImgsubdirpath = pl.path.join(targetRootdir, srcImgsubdirname)
         pl.path.mkdir(targetImgsubdirpath)

         local filenameIdx = srccsvContent.fieldnames:index('Filename')

         -- create scaled images and store in target directory
         local targetdataIdx = 1
         for _, metadata in ipairs(srccsvContent) do
            local srcimgFilename = metadata[filenameIdx]
            local srcimage = image.load(pl.path.join(srcImgsubdir, srcimgFilename))

            for _, trimg in tranformedimages(srcimage, srcimgFilename) do
               local targetimgFilename = trimg[1]
               local transformedimage = trimg[2]
               image.save(pl.path.join(targetImgsubdirpath, targetimgFilename), transformedimage)
               
               local newmetadata = pl.tablex.copy(metadata)
               newmetadata[filenameIdx] = targetimgFilename
               targetdata[targetdataIdx] = newmetadata
               targetdataIdx = targetdataIdx + 1
            end
         end

         -- save csv
         local csvfile = io.open(pl.path.join(targetImgsubdirpath, srccsvFilename),'w')
         pl.data.new(targetdata):write(csvfile)
         io.close(csvfile)
      end
      
      return targetRootdir, srccsvFileprefix
   end

   explodeDataset = function()
      os.execute('unzip '..srcdataset..';')

      local targetTrainRootdir = pl.path.join(targetRootdir, 'Train')
      local targetTestRootdir = pl.path.join(targetRootdir, 'Test')

      pl.path.mkdir(targetRootdir)
      pl.path.mkdir(targetTrainRootdir)
      pl.path.mkdir(targetTestRootdir)

      explode(targetTrainRootdir, pl.path.join(pl.path.currentdir(), srcRootdir, 'Test'))
      explode(targetTestRootdir, pl.path.join(pl.path.currentdir(), srcRootdir, 'Train'))

      return targetRootdir, srccsvFileprefix
   end

end -- do' end


-------------------------------------------------------------------------------------------------------------------
--[[ B. [Optional] Iterate thru the exploded dataset and create feature glipse vectors at predefined locations ]]--
-------------------------------------------------------------------------------------------------------------------
do 
   -- [[ Helper Functions ]]--

   -- Creates a function that will create glimpses for
   -- a given image
   -- Parameters
   -- x and y   - normalized coordinates, (-1, -1) to (1, 1)
   -- gw and gh - width x height of the glimpse window at xy location (pixels)
   -- oh and ow - width x height of the re-sampled glimpse (pixels)
   -- radius    - area from which to generate random glimpses (ratio)
   -- nbr       - number of random glimpse
   -- rseed     - seed to random number generator
   local function glimpser(cfg)
      local x = (cfg.x + 1)/2
      local y = (cfg.y + 1)/2
      local gh = cfg.gh
      local gw = cfg.gw
      local oh = cfg.oh
      local ow = cfg.ow
      local radius = cfg.radius
      local nbr = cfg.nbr
      local rseed = cfg.rseed

      local resampler = nn.SpatialReSampling{oheight = oh, owidth = ow}

      -- creating a shared here saves memory/computation
      local padding = torch.Tensor() -- holds padded image
      local crop = torch.Tensor() -- holds crops of the image thru narrow

      return function(img)
         torch.manualSeed(rseed)

         local output = torch.Tensor(nbr, img:size(1), oh, ow)
         local trX = 0
         local trY = 0

         for sampleIdx = 1, output:size(1) do
            local dst = output[sampleIdx]

            local padheight = math.floor((gh - 2) / 2)
            local padwidth = math.floor((gw - 1) / 2)

            padding:resize(img:size(1), img:size(2) + padheight * 2, img:size(3) + padwidth * 2):zero()
            local center = padding:narrow(2, padheight + 1, img:size(2)):narrow(3, padwidth + 1, img:size(3))
            center:copy(img) -- copying image into padding tensor

            -- cropping
            local h, w = padding:size(2) - gh, padding:size(3) - gw
            local x, y = math.min(h, math.max(0, (x + trX) * h)), math.min(w, math.max(0, (y + trY) * w))

            crop:resize(img:size(1), gh, gw)
            crop:copy(padding:narrow(2, x + 1, gh):narrow(3, y + 1, gw))

            dst:copy(resampler:updateOutput(crop))
            -- image.display(dst) -- after uncommenting use qlua to run the whole program for this lines output
            trX = torch.uniform(-radius, radius)
            trY = torch.uniform(-radius, radius)
         end

         return output
      end
   end

   --[[ Config ]]--

   -- Glimpses to generate
   local features = {}
   features[1] = {
      name = 'collar',
      glimpse = glimpser{
         x = 0,
         y = 0,
         gw = 10,
         gh = 10,
         ow = 48,
         oh = 48,
         radius = 0.05,
         nbr = 10,
         rseed = 100
      }
   }

   local targetRootdir = './item-image-glimpse-dataset'
   local createValidation = true

   --[[ End Config ]]--

   -- Create glimpses
   -- returns
   -- {
   --    'feature_name' = {
   --       numEntries = 0,
   --       data = {
   --          [1] = <Tensor(nbr_glimpses, nbr_imgch = 3, oh, ow)>
   --          [2] = <Tensor(nbr_glimpses, nbr_imgch = 3, oh, ow)>
   --       }
   --    }
   -- }
   local function toGlimpses(srcRootdir, srccsvFileprefix, createValidation)
      local subdirs = pl.dir.getdirectories(srcRootdir)

      local dataset = {}
      local validation = {} -- not used right now

      -- init for each feature
      for _, feature in ipairs(features) do
         dataset[feature.name] = { numEntries = 0, data = {} }
         if createValidation then validation[feature.name] = { numEntries = 0, data = {} } end
      end

      for _, subdir in ipairs(subdirs) do
         local labelcsvFilename = srccsvFileprefix.."-"..(pl.path.basename(subdir)..".csv")
         local csvContent, error = pl.data.read(pl.path.join(subdir, labelcsvFilename))
         local filenameIdx = csvContent.fieldnames:index('Filename')

         local trackforvalidation
         if createValidation then
            local maxtracknbr = 0
            for _, metadata in ipairs(csvContent) do
               local tracknbr = tonumber(metadata[filenameIdx]:split('_')[1])
               if tracknbr > maxtracknbr then maxtracknbr = tracknbr end
            end

            trackforvalidation = math.floor(math.random() * maxtracknbr) + 1
         else
            trackforvalidation = -1
         end


         for _, metadata in ipairs(csvContent) do
            local imgname = metadata[filenameIdx]
            local srcimage = image.load(pl.path.join(subdir, imgname))
            local tracknbr = tonumber(imgname:split("_")[1])

            -- for each of the feature, create glimpse of the image
            for _, feature in ipairs(features) do
               local tgtdataset
               if createValidation and tracknbr == trackforvalidation then tgtdataset = validation
               else tgtdataset = dataset end

               tgtdataset = tgtdataset[feature.name]
               tgtdataset.numEntries = tgtdataset.numEntries + 1
               tgtdataset.data[tgtdataset.numEntries] = feature.glimpse(srcimage)
            end
         end
      end

      if not createValidation then return dataset
      else return dataset, validation end
   end

   createFeatureGlimpse = function (srcRootdir, srccsvFileprefix)
      local srcTrainRootdir = pl.path.join(srcRootdir, 'Train')
      local srcTestRootdir = pl.path.join(srcRootdir, 'Test')
   
      local trainDataset, validationDataset = toGlimpses(srcTrainRootdir, srccsvFileprefix, createValidation)
      local testDataset = toGlimpses(srcTestRootdir, srccsvFileprefix)

      return trainDataset, testDataset, validationDataset
   end

end -- do's end


--------------------------------------------------------------------
--[[ C. [Optional] Persist glimpse vectors in seperate bin file ]]--
--------------------------------------------------------------------
do
   --[[ Config ]]--
   local targetRootdir = './feature-learning-datasets'
   --[[ End Config ]]--

   pl.path.mkdir(targetRootdir)

   --
   local function persist(dataset, nameSuffix)
      for featurename, tbl in pairs(dataset) do
         if tbl.numEntries ~= 0 then
            local featuredataset = {}
            featuredataset.data = torch.Tensor(tbl.numEntries, tbl.data[1]:size(1), tbl.data[1]:size(2), tbl.data[1]:size(3), tbl.data[1]:size(4))
            for idx, t in ipairs(tbl.data) do
               featuredataset.data[idx]:copy(t)
            end
            local filename = featurename.."-glimpses-"..nameSuffix:lower()..".bin"
            local tgtdir = pl.path.join(targetRootdir, featurename)
            pl.path.mkdir(tgtdir)
            torch.save(pl.path.join(tgtdir, filename), featuredataset)
         end
         collectgarbage()
      end
   end 

   persistGlimpses = function (trainDataset, testDataset, validationDataset)
      persist(trainDataset, 'Train')         
      persist(testDataset, 'Test')         
      if validationDataset then persist(trainDataset, 'Validation') end         
   end
end


-------------------------------------------------------------------------------------------------------------------
--[[ D. [Optional] Train spatial transformer network separately for each collection of feature glimpse vectors ]]--
-------------------------------------------------------------------------------------------------------------------
do
   trainModel = function () end
end


-- Final execution
-- [TO DO] Add optional executions
persistGlimpses(createFeatureGlimpse(explodeDataset()))