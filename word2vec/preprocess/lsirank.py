#!/usr/env python

import time
import os
import sys
import logging
import re
import traceback

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from gensim.corpora.dictionary import Dictionary
from gensim.corpora import MmCorpus
from gensim.models import TfidfModel
from gensim.models import LsiModel
from gensim import similarities

try:
	import cpickle as pickle
except:
	import pickle

try:
	import xml.etree.cElementTree as et
except:
	import xml.etree.ElementTree as et


class Lsirank:

	def __init__(self):
		program = os.path.basename(sys.argv[0])
		self.logger = logging.getLogger(program)
		
		#train file paths
		self.trainFilePath = "../datafiles/lsidata/traindata/trainwikiparsed.xml"
		self.trainLemmatizedTextFilePath = "../datafiles/lsidata/traindata/trainwikiparsedlemmatized"
		self.trainPlainTextFilePath = "../datafiles/lsidata/traindata/trainwikiparsedplain"
		self.trainDocCorpusTfidfFilePath = "../datafiles/lsidata/traindata/traindoccorpustfidf.mm"
		self.trainIndexToTitleMapFilePath = "../datafiles/lsidata/traindata/trainindextotitlemap.pkl"

		#model file paths
		self.wordDictFilePath  = "../datafiles/lsidata/model/worddict.dict"
		self.lsiModelFilePath = "../datafiles/lsidata/model/lsiModel.lsi"
		self.lsiIndexFilePath = "../datafiles/lsidata/model/lsiIndex.index"

		#test file paths
		self.testFilePath = "../datafiles/lsidata/testdata/testwikiparsed.xml"
		self.testLemmatizedTextFilePath = "../datafiles/lsidata/testdata/testwikiparsedlemmatized"
		self.testPlainTextFilePath = "../datafiles/lsidata/testdata/testwikiparsedplain"
		self.testDocCorpusTfidfFilePath = "../datafiles/lsidata/testdata/testdoccorpustfidf.mm"
		self.testIndexToTitleMapFilePath = "../datafiles/lsidata/testdata/testindextotitlemap.pkl"

		#result ile paths
		self.testDocRanksFilePath = "../datafiles/lsidata/resultdata/testDocRanks"

		# initialize logger
		logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
		logging.root.setLevel(level=logging.INFO)
		self.logger.info("running %s" % ' '.join(sys.argv))


	def generatePlainTextAndTitleIndexMap(self, readFilePath, writeFilePath, mapFilePath):
		self.logger.info("Generate plain text begin")

		try:
			readFileIter = open(readFilePath, 'r')
			writeFileIter = open(writeFilePath, 'w+')
			indexToTitleMap = {}
			index = 0

			iterparser = et.iterparse(readFileIter, events=('start','end'))
			_, root = iterparser.next()

			for event, element in iterparser:
				if event == 'start':
					continue

				# Start event is required to get the root of the tree
				#print element.tag
				if element.tag == 'title':
					indexToTitleMap[index] = element.text.strip()
					index = index + 1
					pagePlain = ''
					pagePlain = element.text + ' '

				if element.tag == 'text':
					pagePlain += element.text

					#keep only alphabets
					pagePlain = re.sub('[^a-zA-Z]+', ' ', pagePlain)
					pagePlain = pagePlain.lower()
					#print pagePlain
					#print type(pagePlain)
					writeFileIter.write(pagePlain + '\n')
					root.clear()  #clear root to remove parsed part and reclaim memory

			mapFileIter = open(mapFilePath, 'wb+')
			pickle.dump(indexToTitleMap, mapFileIter)
		
		except IOError as e:
			self.logger.info("IOError: %s: %s" % (e.filename, e.strerror))
			raise e
		except Exception as e:
			exc_type, exc_value, exc_traceback = sys.exc_info()
			print exc_type
			print exc_value
			print(repr(traceback.format_tb(exc_traceback)))
			print("LINE WHERE EXCEPTION OCCURED : ", exc_traceback.tb_lineno)
			raise e
		finally:
			if readFileIter is not None:
				readFileIter.close()
			if writeFileIter is not None:
				writeFileIter.close()
			if mapFileIter is not None:
				mapFileIter.close()
			self.logger.info("Generate plain text end")


	def lemmatizeText(self, readFilePath, writeFilePath):
		self.logger.info("Lemmatize text begin")

		try:
			readFileIter = open(readFilePath, 'r')
			writeFileIter = open(writeFilePath, 'w+')
			
			#remove stop words
			stopWords = set(stopwords.words('english'))
			stopWords.add('ref')
			stopWords.add('jpg')
			lemmatizer = WordNetLemmatizer()

			for line in readFileIter.readlines():
				tokenized = word_tokenize(line)
				filteredStopWordsTokens = [word for word in tokenized if not word in stopWords]
				filteredSmallWordsTokens = [word for word in filteredStopWordsTokens if len(word) > 2]
				lammetizedTokens = [lemmatizer.lemmatize(word) for word in filteredSmallWordsTokens]
				for word in lammetizedTokens:
					writeFileIter.write(word + ' ')
				writeFileIter.write('\n')

		except IOError as e:
			self.logger.info("IOError: %s: %s" % (e.filename, e.strerror))
			raise e
		except Exception as e:
			exc_type, exc_value, exc_traceback = sys.exc_info()
			print exc_type
			print exc_value
			print(repr(traceback.format_tb(exc_traceback)))
			print("LINE WHERE EXCEPTION OCCURED : ", exc_traceback.tb_lineno)
			raise e
		finally:
			if readFileIter is not None:
				readFileIter.close()
			if writeFileIter is not None:
				writeFileIter.close()
			self.logger.info("Lemmatize text end")


	def generateDictionary(self, readFilePath, dictFilePath):
		self.logger.info("Generate dictionary and corpus begin")

		try:
			#generate dictionary
			wordDict = Dictionary(self.__getTokensFromFile(readFilePath))
			wordDict.save(dictFilePath)

		except IOError as e:
			self.logger.info("IOError: %s: %s" % (e.filename, e.strerror))
			raise e
		except Exception as e:
			exc_type, exc_value, exc_traceback = sys.exc_info()
			print exc_type
			print exc_value
			print(repr(traceback.format_tb(exc_traceback)))
			print("LINE WHERE EXCEPTION OCCURED : ", exc_traceback.tb_lineno)
			raise e
		finally:
			self.logger.info("Generate dictionary and corpus end")


	def generateTfidfCorpus(self, dictFilePath, readFilePath, tfidfCorpusFilePath):
		self.logger.info("Generate tfidf corpus begin")

		try:
			#generate tfidf vectors
			wordDict = Dictionary.load(dictFilePath)

			#generate bow corpus
			docCorpusBow = [wordDict.doc2bow(docTokens) for docTokens in self.__getTokensFromFile(readFilePath)]

			#generate tfidf corpus
			docModelTfidf = TfidfModel(docCorpusBow, id2word=wordDict,
				normalize=True)
			MmCorpus.serialize(tfidfCorpusFilePath, docModelTfidf[docCorpusBow])

		except IOError as e:
			self.logger.info("IOError: %s: %s" % (e.filename, e.strerror))
			raise e
		except Exception as e:
			exc_type, exc_value, exc_traceback = sys.exc_info()
			print exc_type
			print exc_value
			print(repr(traceback.format_tb(exc_traceback)))
			print("LINE WHERE EXCEPTION OCCURED : ", exc_traceback.tb_lineno)
			raise e
		finally:
			self.logger.info("Generate tfidf corpus end")


	def generateLsiModelAndIndex(self, dictFilePath, tfidfCorpusFilePath, lsiModelFilePath, 
		lsiIndexFilePath):
		self.logger.info("Generate lsi model and index begin")

		try:
			wordDict = Dictionary.load(dictFilePath)
			docCorpusTfidf = MmCorpus(tfidfCorpusFilePath)

			#generate lsi model and indexes
			docModelLsi = LsiModel(corpus=docCorpusTfidf, id2word=wordDict, num_topics=50)
			#TODO: check for Similarity as matrix similarity can take huge space
			docModelLsiIndex = similarities.MatrixSimilarity(docModelLsi[docCorpusTfidf]) 

			#save model and index
			docModelLsi.save(lsiModelFilePath)
			docModelLsiIndex.save(lsiIndexFilePath)

		except IOError as e:
			self.logger.info("IOError: %s: %s" % (e.filename, e.strerror))
			raise e
		except Exception as e:
			exc_type, exc_value, exc_traceback = sys.exc_info()
			print exc_type
			print exc_value
			print(repr(traceback.format_tb(exc_traceback)))
			print("LINE WHERE EXCEPTION OCCURED : ", exc_traceback.tb_lineno)
			raise e
		finally:
			self.logger.info("Generate lsi model ans index end")


	def getLsiRankForDocs(self, tfidfCorpusFilePath, lsiModelFilePath, lsiIndexFilePath,
		rankFilePath, trainTitleMapFilePath, testTitleMapFilePath):
		self.logger.info("Get lsi rank begin")

		try:
			docCorpusTfidf = MmCorpus(tfidfCorpusFilePath)
			docLsiMode = LsiModel.load(lsiModelFilePath)
			docLsiIndex = similarities.MatrixSimilarity.load(lsiIndexFilePath)
			lsiRankMap = {}

			#get lsi vectors for test docs
			docCorpusLsi = docLsiMode[docCorpusTfidf]

			#get similarity score
			simScore = docLsiIndex[docCorpusLsi]

			#load title maps
			trainIndexToTitleMap = pickle.load(open(trainTitleMapFilePath, 'rb'))
			testIndexToTitleMap = pickle.load(open(testTitleMapFilePath, 'rb'))

			#save similarity score for further analysis
			#dimes of sim score is num Test Titles X num Train Titles
			rankFileIter = open(rankFilePath, 'w+')
			testIndex = 0
			for numpyScoreArray in list(simScore):
				scoreList = numpyScoreArray.tolist()
				maxSimScore = max(scoreList)
				maxScoreIndex = scoreList.index(maxSimScore)
				lsiRankMap[testIndexToTitleMap[testIndex]] = (trainIndexToTitleMap[maxScoreIndex], maxSimScore)
				testIndex = testIndex + 1

			for link, seedscoretuple in sorted(lsiRankMap.items(), key = lambda x:x[1][1], reverse = True):
				rankFileIter.write(link + ' | ' + seedscoretuple[0] + ' | ' + str(seedscoretuple[1]) + '\n')

		except IOError as e:
			self.logger.info("IOError: %s: %s" % (e.filename, e.strerror))
			raise e
		except Exception as e:
			exc_type, exc_value, exc_traceback = sys.exc_info()
			print exc_type
			print exc_value
			print(repr(traceback.format_tb(exc_traceback)))
			print("LINE WHERE EXCEPTION OCCURED : ", exc_traceback.tb_lineno)
			raise e
		finally:
			if rankFileIter is not None:
				rankFileIter.close()
			self.logger.info("Get lsi rank end")


	def trainLsiModel(self):
		self.logger.info("Train lsi model begin")
		try:
			#generate plain text from xml file
			self.generatePlainTextAndTitleIndexMap(self.trainFilePath, self.trainPlainTextFilePath,
				self.trainIndexToTitleMapFilePath)
			#lammetize text
			self.lemmatizeText(self.trainPlainTextFilePath, self.trainLemmatizedTextFilePath)
			#generate word dictionary
			self.generateDictionary(self.trainLemmatizedTextFilePath, self.wordDictFilePath)
			#generate tfidf corpus from taining data
			self.generateTfidfCorpus(self.wordDictFilePath, self.trainLemmatizedTextFilePath,
				self.trainDocCorpusTfidfFilePath)
			#generate lsi model and indexes
			self.generateLsiModelAndIndex(self.wordDictFilePath, self.trainDocCorpusTfidfFilePath,
				self.lsiModelFilePath, self.lsiIndexFilePath)

		except Exception as e:
			raise e
		finally:
			self.logger.info("Train lsi model end")


	def testLsiModel(self):
		self.logger.info("Test lsi model begin")
		try:
			#generate plain text from xml file
			self.generatePlainTextAndTitleIndexMap(self.testFilePath, self.testPlainTextFilePath,
				self.testIndexToTitleMapFilePath)
			#lammetize text
			self.lemmatizeText(self.testPlainTextFilePath, self.testLemmatizedTextFilePath)
			#generate tfidf corpus from test data
			self.generateTfidfCorpus(self.wordDictFilePath, self.testLemmatizedTextFilePath,
				self.testDocCorpusTfidfFilePath)
			#generate lsi model and indexes
			self.getLsiRankForDocs(self.testDocCorpusTfidfFilePath, self.lsiModelFilePath, 
				self.lsiIndexFilePath, self.testDocRanksFilePath, self.trainIndexToTitleMapFilePath,
				self.testIndexToTitleMapFilePath)

		except Exception as e:
			raise e
		finally:
			self.logger.info("Test lsi model end")


	def __getTokensFromFile(self, readFilePath):
		self.logger.info("Get tokens from file begin")

		try:
			readFileIter = open(readFilePath, 'r')
			for line in readFileIter.readlines():
				tokenized = [word for word in line.split()]
				yield tokenized

		except IOError as e:
			self.logger.info("IOError: %s: %s" % (e.filename, e.strerror))
			raise e
		except Exception as e:
			exc_type, exc_value, exc_traceback = sys.exc_info()
			print exc_type
			print exc_value
			print(repr(traceback.format_tb(exc_traceback)))
			print("LINE WHERE EXCEPTION OCCURED : ", exc_traceback.tb_lineno)
			raise e
		finally:
			if readFileIter is not None:
				readFileIter.close()
			self.logger.info("Get tokens from file end")


if __name__ == '__main__':
	lsirank = Lsirank()
	lsirank.trainLsiModel()
	lsirank.testLsiModel()
