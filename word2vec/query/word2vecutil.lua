local pl = (require 'pl.import_into')()

torch.setdefaulttensortype('torch.FloatTensor')

options = {
	vectorFilePath = '../datafiles/word2vecdata/nounphrasevectors.bin',
	vectorTorchFilePath = '../datafiles/word2vecdata/nounphrasevectors.t7'
}

local word2vec = {}

print('Loading word2vec data')
if not paths.filep(options.vectorTorchFilePath) then
	word2vec = require('binvectortotorch')
else
	word2vec = torch.load(options.vectorTorchFilePath)
end
print('Loaded word2vec data')



-- get similar words
function word2vec:getSimilarWords(word, numSimilarWords)
	local numSimilarWords = numSimilarWords or 5

	local vector = word2vec:getVector(word)
	if vector == nil then
		--word not found
		return nil
	end

	local norm = vector:norm(2)
	vector:div(norm)

	local cosDistances = torch.mv(self.vectorMatrix, vector)
	cosDistances, oldVectorIndexes = torch.sort(cosDistances, 1, true)

	local nearestWords = {}
	local similarityScore = {}

	for i = 1, numSimilarWords do
		table.insert(nearestWords, self.vocabIndexToWord[oldVectorIndexes[i]])
		table.insert(similarityScore, cosDistances[i])
	end

	return {nearestWords, similarityScore}
end

-- get similar word vectors
function word2vec:getSimilarWordVectors(word, numSimilarWords)
	local numSimilarWords = numSimilarWords or 5

	local vector = word2vec:getVector(word)
	if vector == nil then
		--word not found
		return nil
	end

	local norm = vector:norm(2)
	vector:div(norm)

	local cosDistances = torch.mv(self.vectorMatrix, vector)
	cosDistances, oldVectorIndexes = torch.sort(cosDistances, 1, true)

	
	local nearestWordVectors = {}
	local similarityScore = {}

	for i = 1, numSimilarWords do
		table.insert(nearestWordVectors, self.vectorMatrix[oldVectorIndexes[i]])
		table.insert(similarityScore, cosDistances[i])
	end

	return {nearestWordVectors, similarityScore}
end

-- vector query
function word2vec:getVector(word)
	local wordIndex = self.wordToVocabIndex[word]
	if wordIndex == nil then
		-- word not found
		return nil
	end
	return self.vectorMatrix[wordIndex]
end

-- get all keys
function word2vec:getKeywords()
   for k, _ in pairs(self.wordToVocabIndex) do
   	print(k)
   end
end

return word2vec