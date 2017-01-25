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


function loadModel(file)
	local vectorFile = torch.DiskFile(file, 'r')
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

	return wordToVocabIndex, vocabIndexToWord, vectorMatrix
end

word2vec = {}
local wordToVocabIndex
local vocabIndexToWord 
local vectorMatrix

-- load noun model
wordToVocabIndex, vocabIndexToWord, vectorMatrix = loadModel(options.nounVectorFilePath)
word2vec.nounVectorMatrix = vectorMatrix
word2vec.nounWordToVocabIndex = wordToVocabIndex
word2vec.nounVocabIndexToWord = vocabIndexToWord

-- load adjective model
wordToVocabIndex, vocabIndexToWord, vectorMatrix = loadModel(options.adjVectorFilePath)
word2vec.adjVectorMatrix = vectorMatrix
word2vec.adjWordToVocabIndex = wordToVocabIndex
word2vec.adjVocabIndexToWord = vocabIndexToWord

torch.save(options.torchVectorsFilePath, word2vec)
print('binary vectors to torch table conversion complete')

return word2vec