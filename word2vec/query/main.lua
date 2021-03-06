local pl = (require 'pl.import_into')()
stringx.import()

local word2vecutil = require 'word2vecutil'

--execution
local word
local numSimilarWords
local form
while true do
	io.write("Enter query word to find similar words for or exit to stop:\n")
	io.flush()
	word = io.read()
	word = word:strip()

	if word == "exit" then
		-- exit loop
		break
	end

	io.write("Enter number of similar words to get:\n")
	io.flush()
	numSimilarWords = io.read()
	numSimilarWords = tonumber(numSimilarWords)

	io.write("Enter form of words to query, noun or adjective:\n")
	io.flush()
	form = io.read()
	form = form:strip()

	neighbors = word2vecutil:getSimilarWords(word, form, numSimilarWords)
	if neighbors == nil then
		io.write("Word not found in dictionary or queried form is invalid.\n")
	else 
		-- neighbors table contains two tables of similar words
		-- and corresponsing similarity score
		pl.pretty.dump(neighbors)
		print()		
	end
end