local pl = (require 'pl.import_into')()

torch.setdefaulttensortype('torch.FloatTensor')

options = {
	nounVectorFilePath = '../datafiles/word2vecdata/nounphrasevectors.bin',
	adjVectorFilePath = '../datafiles/word2vecdata/adjectivevectors.bin',
	torchVectorsFilePath = '../datafiles/word2vecdata/torchVectors.t7'
}

local word2vec = {}

print('Loading word2vec data')
if not paths.filep(options.torchVectorsFilePath) then
	word2vec = require('binvectortotorch')
else
	word2vec = torch.load(options.torchVectorsFilePath)
end
print('Loaded word2vec data')


-- get similar words
function word2vec:getSimilarWords(word, form, numSimilarWords)
	local numSimilarWords = numSimilarWords or 5

	local wordToVocabIndex
	local vocabIndexToWord
	local vectorMatrix
	if form == 'noun' then
		wordToVocabIndex = self.nounWordToVocabIndex
		vocabIndexToWord = self.nounVocabIndexToWord
		vectorMatrix = self.nounVectorMatrix
	elseif form == 'adjective' then
		wordToVocabIndex = self.adjWordToVocabIndex
		vocabIndexToWord = self.adjVocabIndexToWord
		vectorMatrix = self.adjVectorMatrix
	else 
		-- invalid form 
		return nil
	end

	local vector = word2vec:getVector(word, wordToVocabIndex, vectorMatrix)
	if vector == nil then
		--word not found
		return nil
	end

	local norm = vector:norm(2)
	vector:div(norm)

	local cosDistances = torch.mv(vectorMatrix, vector)
	cosDistances, oldVectorIndexes = torch.sort(cosDistances, 1, true)

	local nearestWords = {}
	local similarityScore = {}

	for i = 1, numSimilarWords do
		table.insert(nearestWords, vocabIndexToWord[oldVectorIndexes[i]])
		table.insert(similarityScore, cosDistances[i])
	end

	return {nearestWords, similarityScore}
end

-- get similar word vectors
function word2vec:getSimilarWordVectors(word, form, numSimilarWords)
	local numSimilarWords = numSimilarWords or 5

	local wordToVocabIndex
	local vocabIndexToWord
	local vectorMatrix
	if form == 'noun' then
		wordToVocabIndex = self.nounWordToVocabIndex
		vocabIndexToWord = self.nounVocabIndexToWord
		vectorMatrix = self.nounVectorMatrix
	elseif form == 'adjective' then
		wordToVocabIndex = self.adjWordToVocabIndex
		vocabIndexToWord = self.adjVocabIndexToWord
		vectorMatrix = self.adjVectorMatrix
	else 
		-- invalid form 
		return nil
	end

	local vector = word2vec:getVector(word, wordToVocabIndex, vectorMatrix)
	if vector == nil then
		--word not found
		return nil
	end

	local norm = vector:norm(2)
	vector:div(norm)

	local cosDistances = torch.mv(vectorMatrix, vector)
	cosDistances, oldVectorIndexes = torch.sort(cosDistances, 1, true)

	
	local nearestWordVectors = {}
	local similarityScore = {}

	for i = 1, numSimilarWords do
		table.insert(nearestWordVectors, vectorMatrix[oldVectorIndexes[i]])
		table.insert(similarityScore, cosDistances[i])
	end

	return {nearestWordVectors, similarityScore}
end

-- vector query
function word2vec:getVector(word, wordToVocabIndex, vectorMatrix)
	local wordIndex = wordToVocabIndex[word]
	if wordIndex == nil then
		-- word not found
		return nil
	end
	return vectorMatrix[wordIndex]
end

-- get all keys for a vocab 
function word2vec:getKeywords(wordToVocabIndex)
   for k, _ in pairs(wordToVocabIndex) do
   	print(k)
   end
end

return word2vec