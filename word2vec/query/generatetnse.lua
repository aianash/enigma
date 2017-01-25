require 'torch'

local vectorTorchFilePath = '../datafiles/word2vecdata/nounphrasevectors.t7'


-- function that draw text map: [modified from manifold]
function drawTextMap(mappedData, words, inpMapSize, inpFontSize)
  -- this function assumes embeddings normalized to 1
  -- input options:
  local mapSize  = inpMapSize or 2048
  local fontSize = inpFontSize or 10
  
  -- check inputs are correct:
  local N = mappedData:size(1)
  if mappedData:nDimension() ~= 2 or mappedData:size(2) ~= 2 then
    error('This function is designed to operate on 2D embeddings only.')
  end
  if mappedData:size(1) ~= #words then
    error('Number of words should match the number of rows in X.')
  end
    
    -- normalize data 
    mappedData:add(-mappedData:min())
    mappedData:div(mappedData:max())
  
  -- prepare image for rendering:
  require 'image'
  require 'qtwidget'
  require 'qttorch'
  local win = qtwidget.newimage(mapSize, mapSize)
  
  --render the words:
  for key, val in pairs(words) do
    win:setfont(qt.QFont{serif = false, size = fontSize})
    win:moveto(math.floor(mappedData[key][1] * mapSize), math.floor(mappedData[key][2] * mapSize))
    win:show(val)
  end

   -- render to tensor:
  local mapImage = win:image():toTensor()
  
  -- return text map:
  return mapImage
end


-- show map with original digits:
local function showTextMap(mappedData, vocab)
  -- draw map with words:
  local textMap = drawTextMap(mappedData, vocab)
  -- plot results:
  local display = require 'display'
  display.image(textMap)
end


-- function that generate t-SNE plot for word2vec:
local function generateTsne()
  -- amount of data to use for display:
  local numTestData = 1000
  -- load subset of word2vec to diaplay:
  local word2vec = torch.load(vectorTorchFilePath)
  local data = word2vec.vectorMatrix:narrow(1, 1, numTestData)
  data = torch.DoubleTensor(data:size()):copy(data)

  -- create vocab of numTestData words
  local vocab = {}
  for i = 1, numTestData do
    vocab[i] = word2vec.vocabIndexToWord[i]
  end

  local manifold = require 'manifold'
  -- run t-SNE:
  local timer = torch.Timer()
  opts = {ndims = 2, perplexity = 30, pca = 20, use_bh = false}
  mappedData = manifold.embedding.tsne(data, opts)
  print('Successfully performed t-SNE in ' .. timer:time().real .. ' seconds.')
  showTextMap(mappedData, vocab)

  -- -- run Barnes-Hut t-SNE:
  -- opts = {ndims = 2, perplexity = 30, pca = 50, use_bh = true, theta = 0.5}
  -- timer:reset()
  -- mapped_x2 = manifold.embedding.tsne(x, opts)
  -- print('Successfully performed Barnes Hut t-SNE in ' .. timer:time().real .. ' seconds.')
  -- show_map(mapped_x2, x:clone())
end


-- run the demo:
generateTsne()

