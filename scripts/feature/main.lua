local pl = (require 'pl.import_into')()
require 'image'
require 'torch'
require 'nnx'
require 'enigma'
require 'distributions'
require 'gnuplot'
require 'socket'

pl.stringx.import()
torch.setdefaulttensortype('torch.FloatTensor')

function sleep(sec)
   socket.select(nil, nil, sec)
end

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

      print(string.format([[
-----------------
Exploding Dataset
-----------------
zip = %s]], srcdataset))

      local targetTrainRootdir = pl.path.join(targetRootdir, 'Train')
      local targetTestRootdir = pl.path.join(targetRootdir, 'Test')

      pl.path.mkdir(targetRootdir)
      pl.path.mkdir(targetTrainRootdir)
      pl.path.mkdir(targetTestRootdir)

      explode(targetTrainRootdir, pl.path.join(pl.path.currentdir(), srcRootdir, 'Train'))
      explode(targetTestRootdir, pl.path.join(pl.path.currentdir(), srcRootdir, 'Test'))

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
         gw = 16,
         gh = 16,
         ow = 10,
         oh = 10,
         radius = 0.02,
         nbr = 10,
         rseed = 100
      }
   }

   -- local targetRootdir = './item-image-glimpse-dataset'
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
            --                                  numEntries     numGlimpses          image channel        height               width
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
      return targetRootdir
   end
end


-------------------------------------------------------------------------------------------------------------------
--[[ D. [Optional] Train spatial transformer network separately for each collection of feature glimpse vectors ]]--
-------------------------------------------------------------------------------------------------------------------
do
   --[[ Config ]]--
   local oH = 32
   local oW = 32

   local pretrainEpoch = 4
   local epoch = 10
   local batchSize = 2

   local targetModeldir = './feature-models'
   --[[ End Config ]]--

   -- local models = require 'scripts.feature.models'
   -- local mobius = require 'scripts.feature.mobius'

   trainModel = function (datasetRootdir)
      local featuredirs = pl.dir.getdirectories(datasetRootdir)
      table.sort(featuredirs)

      for _, featuredir in ipairs(featuredirs) do
         local featurename = pl.path.basename(featuredir)

         local trainDataset = torch.load(pl.path.join(featuredir, featurename.."-glimpses-train.bin"))
         local testDataset = torch.load(pl.path.join(featuredir, featurename.."-glimpses-test.bin"))
         local validationDataset = torch.load(pl.path.join(featuredir, featurename.."-glimpses-validation.bin")) -- [TO DO] test if present first

         local CMFA = require 'scripts.feature.cmfa'
         local HeuristicCMFA = require 'scripts.feature.heuristiccmfa'

         local run = "exp3"

         -- Dummy Experiment 1
         if run == "exp1" then
            -- generate dataset
            function generate(n, p, k, f, Psi, offset)
               local X_star = torch.Tensor(n, f)
               local L = torch.randn(p, k)
               local G = torch.randn(p, f)
               local E_starI = torch.randn(f, f)

               local Cov = L * L:t() + G * E_starI * G:t() + Psi
               local Y = torch.Tensor(n, p)

               local za = torch.rand(k)
               local xa = torch.rand(f)

               for i = 1, n do
                  -- local x = distributions.dir.rnd(xa):float() + offset
                  local x = torch.randn(f) + offset
                  local z = distributions.dir.rnd(za):float()
                  X_star[i] = x
                  local mean = G * x + L * z
                  Y[i] = distributions.mvn.rnd(mean, Cov)
               end

               return Y:t(), X_star:t()
            end

            local N = 4000
            local n = 1000
            local s = N / n
            local p = 2
            local k = 1
            local f = 1

            local Y = torch.zeros(p, N)
            local X_star = torch.zeros(f, N)

            local Psi = torch.diag(torch.randn(p))

            for batchIdx = 1, (N / n) do
               y, x_star = generate(n, p, k, f, Psi, batchIdx * 3)
               Y:narrow(2, (batchIdx - 1) * n + 1, n):copy(y)
               X_star:narrow(2, (batchIdx - 1) * n + 1, n):copy(x_star)
            end

            local model = CMFA:new{
               batchSize = N, -- no batch
               numComponents = 3,
               outputVectorSize = p,
               factorVectorSize = k,
               inputVectorSize = f,
               datasetSize = N,
               delay = 1,
               forgettingRate = 0.6
            }

            local Gm, Gcov, Lm, Lcov, Zm, Xm, Qs = model:train(Y, X_star, 12)

            local N1 = 0
            local N2 = 0
            local N3 = 0

            for i = 1, N do
               local qs = Qs[i]
               local s
               if qs[1] >= qs[2] and qs[1] >= qs[3] then N1 = N1 + 1
               elseif qs[2] >= qs[1] and qs[2] >= qs[3] then N2 = N2 + 1
               else N3 = N3 + 1 end
            end

            local Y1new = torch.zeros(p, N1)
            local X1 = torch.Tensor(f, N1)

            local Y2new = torch.zeros(p, N2)
            local X2 = torch.Tensor(f, N2)

            local Y3new = torch.zeros(p, N3)
            local X3 = torch.Tensor(f, N3)

            local i1 = 1
            local i2 = 1
            local i3 = 1

            for i = 1, N do
               local qs = Qs[i]
               local s
               if qs[1] >= qs[2] and qs[1] >= qs[3] then s = 1
               elseif qs[2] >= qs[1] and qs[2] >= qs[3] then s = 2
               else s = 3 end

               z = Zm[{s, {}, i}]
               x = Xm[{s, {}, i}]

               if s == 1 then
                  Y1new[{ {}, i1 }] = Lm[s] * z + Gm[s] * x
                  X1[{ {}, i1 }] = X_star[{ {}, i }]
                  i1 = i1 + 1
               elseif s == 2 then
                  Y2new[{ {}, i2 }] = Lm[s] * z + Gm[s] * x
                  X2[{ {}, i2 }] = X_star[{ {}, i }]
                  i2 = i2 + 1
               else
                  Y3new[{ {}, i3 }] = Lm[s] * z + Gm[s] * x
                  X3[{ {}, i3 }] = X_star[{ {}, i }]
                  i3 = i3 + 1
               end
            end

            gnuplot.figure(1)
            gnuplot.scatter3({'true', Y[1], Y[2], X_star:squeeze()})

            gnuplot.figure(2)
            gnuplot.scatter3({'predicted1', Y1new[1], Y1new[2], X1:squeeze()},
               {'predicted2', Y2new[1], Y2new[2], X2:squeeze()})

            gnuplot.figure(3)
            gnuplot.scatter3({'predicted2', Y2new[1], Y2new[2], X2:squeeze()},
               {'predicted3', Y3new[1], Y3new[2], X3:squeeze()})

            gnuplot.figure(4)
            gnuplot.scatter3({'predicted3', Y3new[1], Y3new[2], X3:squeeze()},
               {'predicted1', Y1new[1], Y1new[2], X1:squeeze()})

            gnuplot.figure(5)
            gnuplot.scatter3({'predicted1', Y1new[1], Y1new[2], X1:squeeze()},
                            {'predicted2', Y2new[1], Y2new[2], X2:squeeze()},
                            {'predicted3', Y3new[1], Y3new[2], X3:squeeze()})

            io.read()
         end


         -- Dummy experitment 2
         if run == "exp2" then
            local outputVectorSize = 50
            local datasetSize = 400
            local inputVectorSize = 8
            local factorVectorSize = 4
            local numComponents = 2

            local model = CMFA:new{
               batchSize = 400,
               numComponents = numComponents,
               outputVectorSize = outputVectorSize,
               factorVectorSize = factorVectorSize,
               inputVectorSize = inputVectorSize, -- 16 x 16 inout image
               datasetSize = datasetSize,
               delay = 1,
               forgettingRate = 0.6
            }

            local X_star = torch.randn(datasetSize, inputVectorSize)

            local Y = torch.Tensor(datasetSize, outputVectorSize)

            local G1 = torch.rand(outputVectorSize, inputVectorSize) * 20
            local L1 = torch.rand(outputVectorSize, factorVectorSize) * 20

            local G2 = torch.rand(outputVectorSize, inputVectorSize)
            local L2 = torch.rand(outputVectorSize, factorVectorSize)

            for i = 1, datasetSize do
               local z = torch.randn(factorVectorSize)

               if i % 3 == 0 then
                  Y[i] = G1 * X_star[i] + L1 * z + torch.randn(outputVectorSize)
               else Y[i] = G2 * X_star[i] + L2 * z + torch.randn(outputVectorSize) end
            end

            print(string.format([[
   ---------------------------
   Dataset
   ---------------------------
   L =
   %s

   G =
   %s
   ]], L, G))

            local Gm, Gcov, Lm, Lcov = model:train(Y:t(), X_star:t(), 30)

            print(Gm)
            -- print(G1)

            print(Gcov)

            print(Lm)
            -- print(L1)

            print(Lcov)
            print(torch.dist(Gm[1], G1))
            print(torch.dist(Lm[1], L1))
         end

         if run == "exp3" then
            local I = trainDataset.data
            local M, g, c, h, w = I:size(1), I:size(2), I:size(3), I:size(4), I:size(5)
            local N = M * g
            local h_star, w_star = 3, 3

            -- normalize dataset
            function normalize(data)
               local normKernel = image.gaussian1D(7)
               local norm = nn.SpatialContrastiveNormalization(3, normKernel)
               local batchSize = 200
               local N = data:size(1)
               for i = 1, N, batchSize do
                  local batch = math.min(N, i + batchSize) - i
                  local nImgs = norm:forward(data:narrow(1, i, batch))
                  data:narrow(1, i, batch):copy(nImgs)
               end
            end

            normalize(I:view(M * g, c, h, w))

            Y = torch.zeros(h * w, N)
            X_star = torch.zeros(h_star * w_star, N)

            print(string.format("M = %d g = %d", M, g))

            for i = 1, M do
               for j = 1, g do
                  local n = (i - 1) * g + j
                  local src = I[{i, j, {}, {}, {}}]
                  local grey = image.rgb2y(src)

                  image.save("./glimpses/"..tostring(n).."-1.ppm", src)
                  image.save("./glimpses/"..tostring(n).."-2.pgm", grey)

                  Y[{ {}, n }]:copy(grey)
                  X_star[{ {}, n }]:copy(image.scale(grey, w_star, h_star))
               end
            end

            print(string.format("N = %d", N))
            local s = 2
            local k = 2
            local p = h * w
            local f = h_star * w_star

            local cmfa = CMFA:new{
               batchSize = N,
               numComponents = s,
               outputVectorSize = p,
               factorVectorSize = k,
               inputVectorSize = f, -- 16 x 16 inout image
               datasetSize = N,
               delay = 1,
               forgettingRate = 0.6
            }

            local heuristic = HeuristicCMFA:new(cmfa)

            cmfa:check(Y, "Yn")
            local Gm, Gcov, Lm, Lcov, Zm, Xm, Qs = cmfa:train(Y, X_star, 13)

            heuristic:train(Xm, Zm, X_star)
            local hGx, hLz = heuristic:forward(X_star)
            local heuristicY = hGx --+ hLz -- s x p x n

            if pl.path.isdir("./output") then pl.dir.rmtree("./output") end
            pl.path.mkdir("./output")
            pl.path.mkdir("./output/allsep")

            for t = 1, s do
               pl.path.mkdir("./output/"..tostring(t))
            end

            local Ypred = torch.zeros(p, N)

            for t = 1, s do
               local Qst = Qs[{ {}, t }]
               local Lz = Lm[t] * Zm[t]
               local Gx = Gm[t] * Xm[t]
               local Yt = Lz + Gx -- p x n
               local pQst = Qst:contiguous():view(1, N):expand(p, N) -- -- p x n
               local YpQst = torch.cmul(pQst, Yt) -- p x n
               Ypred = Ypred + YpQst

               for i = 1, N do
                  local y = Yt[{ {}, i }]:contiguous():view(h, w)
                  local x = Xm[{t, {}, i}]:contiguous():view(h_star, w_star)
                  local lz = Lz[{ {}, i }]:contiguous():view(h, w)
                  local gx = Gx[{ {}, i }]:contiguous():view(h, w)
                  local hy = heuristicY[{t, {}, i }]:contiguous():view(h, w)

                  image.save("./output/allsep/"..tostring(i).."-"..tostring(t).."-xm.pgm", image.scale(x, w_star * 8, h_star * 8))
                  image.save("./output/allsep/"..tostring(i).."-"..tostring(t).."-y.pgm", image.scale(y, w * 8, h * 8))
                  image.save("./output/allsep/"..tostring(i).."-"..tostring(t).."-Lz.pgm", image.scale(lz, w * 8, h * 8))
                  image.save("./output/allsep/"..tostring(i).."-"..tostring(t).."-Gx.pgm", image.scale(gx, w * 8, h * 8))
                  image.save("./output/allsep/"..tostring(i).."-"..tostring(t).."-yh.pgm", image.scale(hy, w * 8, h * 8))
               end
            end

            for i = 1, N do
               local y = Y[{ {}, i }]:contiguous():view(h, w)
               image.save("./output/allsep/"..tostring(i).."-"..tostring(s + 1).."-target.pgm", image.scale(y, w * 8, h * 8))

               y = Ypred[{ {}, i }]:contiguous():view(h, w)
               image.save("./output/allsep/"..tostring(i).."-"..tostring(s + 2).."-weight.pgm", image.scale(y, w * 8, h * 8))

               local x = X_star[{ {}, i }]:contiguous():view(h_star, w_star)
               image.save("./output/allsep/"..tostring(i).."-"..tostring(s + 3).."-x_star.pgm", image.scale(x, w_star * 8, h_star * 8))
            end

            pl.path.mkdir("./output/G")
            pl.path.mkdir("./output/L")
            for t = 1, s do
               local gt = "./output/G/"..tostring(t)
               local lt = "./output/L/"..tostring(t)
               pl.path.mkdir(gt)
               pl.path.mkdir(lt)

               for j = 1, f do
                  local g = Gm[{t, {}, j}]:contiguous():view(h, w)
                  image.save(gt.."/"..tostring(j)..".pgm", image.scale(g, w * 8, h * 8))
               end

               local ker = image.gaussian({size=9,sigma=1.591/9,normalize=true}):type('torch.DoubleTensor')

               for j = 1, k do
                  local l = Lm[{t, {}, j}]:contiguous():view(h, w):double()
                  image.save(lt.."/"..tostring(j)..".pgm", image.scale(l, w * 8, h * 8))
               end
            end
         end

         -- local gC, gH, gW = trainDataset:size(3), trainDataset:size(4), trainDataset:size(5)

         -- local featureSTM = models.newFeaturemodel()
         -- local featureMFA = models.newMFAmodel(oH, oW)

         -- [TO DO] pre-training independently the MFA model
         -- as its generative. dataset is directly scaled
         -- for e = 1, pretrainEpoch do
         --    featureFA:pretrain(trainDataset)
         -- end

         -- local featureSTMOptState = {
         --    momentum = 2,
         --    learningRate = 1,
         --    learningRateDecay = 1,
         --    weightDecay = 2
         -- }

         -- mobius training
         -- [TO DO] probably no need for primary and secondaris... just branches
         -- local trainer = mobius.MobiusTrainer:new{
         --    topology = { -- defining topology of mobius learning
         --       [1] = {
         --          model = featureSTM,
         --          optimizer = mobius.NNOptim:new(featureSTM, optim.sgd, featureSTMOptState),
         --          parent = {
         --             model = mobius.Identity:new() -- [TODO] this is wrong here...
         --          }
         --       },

         --       [2] = {
         --          model = mobius.Nothing:new(),
         --          parent = {
         --             model = featureMFA,
         --             optimizer = mobius.MFAOptim:new(featureMFA)
         --          }
         --       }
         --    },
         --    iterations = { name = 'exponential-backoff', max = 10, min = 5 },
         --    batchSize = batchSize
         -- }

         -- for e = 1, epoch do
         --    print("running epoch")
         --    trainer:train(trainDataset) -- train on dataset
         --    trainer:resetIterationScheme()

         --    -- calculate accuracy on test
         --    -- persist current model
         -- end

      end
   end
end


-- Final execution
-- [TO DO] Add optional executions
trainModel(persistGlimpses(createFeatureGlimpse(explodeDataset())))