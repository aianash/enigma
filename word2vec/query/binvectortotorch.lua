---------------------------------
-- FILE FORMAT
-- vocabsize (int)
-- vector ize (int)
-- word1(string) vector1(float)
-- word2(string) vector2(float)
-- ....
-- ....
---------------------------------
local pl = (require 'pl.import_into')()
stringx.import()

local vectorFile = torch.DiskFile(options.vectorFilePath, 'r')
local maxWordLength = 100 -- maximum length of vocab strings while training

-- functions
function readWord(file)
	local word = {}
	for i = 1, maxWordLength do
		local readChar = file:readChar()
		if readChar == 32 then	
		 	break
		else
			word[#word + 1] = readChar
		end
	end
	word = torch.CharStorage(word)
	return word:string():strip()
end


-- ascii handler
vectorFile:ascii()

local vocabSize = vectorFile:readInt()
local vectorSize = vectorFile:readInt()

local wordToVocabIndex = {}
local vocabIndexToWord = {}
local vectorMatrix = torch.FloatTensor(vocabSize, vectorSize)

-- read vectors as binary
vectorFile:binary()

for i = 1, vocabSize do
	local word = readWord(vectorFile)
	--print(word)
	local vector = vectorFile:readFloat(vectorSize)
	vector = torch.FloatTensor(vector)

	local norm = torch.norm(vector, 2)
	if norm ~= 0 then
		vector:div(norm)
	end

	wordToVocabIndex[word] = i
	vocabIndexToWord[i] = word

	vectorMatrix[{{i}, {}}] = vector
end

-- write torch table to file
word2vec = {}
word2vec.vectorMatrix = vectorMatrix
word2vec.wordToVocabIndex = wordToVocabIndex
word2vec.vocabIndexToWord = vocabIndexToWord

torch.save(options.vectorTorchFilePath, word2vec)
print('binary vectors to torch table conversion complete')

return word2vec