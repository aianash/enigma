#!/usr/env python

import nltk
import re
import time
import os
import sys
import logging
import traceback

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

try:
	import xml.etree.cElementTree as et
except ImportError:
	import xml.etree.ElementTree as et


class Preprocessor:

	def __init__(self):
		program = os.path.basename(sys.argv[0])
		self.logger = logging.getLogger(program)
		self.pageoutputFilePath = "../datafiles/preprocesseddata/pageoutput.xml"
		self.plaintextFilePath = "../datafiles/preprocesseddata/plaintextoutput"
		self.processedFilePath = "../datafiles/preprocesseddata/taggedwords"
		self.nounphrasesFilePath = "../datafiles/preprocesseddata/nounphrases"
		self.lemmatizedTextFilePath = "../datafiles/preprocesseddata/lemmatizedText"

		# initialize logger
		logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
		logging.root.setLevel(level=logging.INFO)
		self.logger.info("running %s" % ' '.join(sys.argv))

	def preprocess(self):
		self.logger.info("Preprocessing begin")

		try:
			# open content file
			self.__generate_plain_text()
			self.__lemmatize_text()
			self.__extract_tags()
			self.__get_noun_phrases_and_adjectives()

		except Exception as e:
			self.logger.info(str(e))

		self.logger.info("Preprocessing end")


	def __get_noun_phrases(self):
		self.logger.info("Get noun phrases begin")

		try:
			processedFileIter = open(self.processedFilePath, 'r')
			nounphrasesFileIter = open(self.nounphrasesFilePath, 'wb+')

			#get noun phrases, combine words that are NN,NNS,NNP,NNPS and are 
			#contiguous, Break on full stop
			nounPhraseBegin = False
			nounPhrase = ''
			while True:
				line = processedFileIter.readline()
				if not line: #end of file
					if nounPhraseBegin == True:
						nounphrasesFileIter.write(nounPhrase) 
					break

				word, tag = line.split()
				#print word, tag
				word, tag = word.strip().lower(), tag.strip()
				#print repr(tag)

				# if tag == 'NN':
				# 	print 'found NN'
				#skip till we found a noun
				if nounPhraseBegin == False and (tag != 'NN' and tag != 'NNS' and tag != 'NNP' and tag != 'NNPS'):
					#print 'not found'
					continue

                # start of a noun phrase
				if nounPhraseBegin == False:
					#print 'found'
					nounPhraseBegin = True
				 	nounPhrase = word
				 	continue

				#continue till we find nouns
				if tag == 'NN' or tag == 'NNS' or tag == 'NNP' or tag == 'NNPS':
					nounPhrase += '_' + word
					continue

				#end of noun phrase
				nounPhraseBegin = False
				nounphrasesFileIter.write(nounPhrase + ' ')


		except IOError as e:
			self.logger.info("IOError: %s: %s" % (e.filename, e.strerror))
		finally:
			if processedFileIter is not None:
				processedFileIter.close()
			if nounphrasesFileIter is not None:
				nounphrasesFileIter.close()
			self.logger.info("Get noun phrases end")


	def __get_noun_phrases_and_adjectives(self):
		self.logger.info("Get noun phrases begin")

		try:
			readFileIter = open(self.processedFilePath, 'r')
			writeFileIter = open(self.nounphrasesFilePath, 'wb+')

			#get noun phrases, combine words that are NN,NNS,NNP,NNPS and are 
			#contiguous, Break on full stop
			nounPhraseBegin = False
			nounPhrase = ''
			while True:
				line = readFileIter.readline()
				if not line: #end of file
					if nounPhraseBegin == True:
						writeFileIter.write(nounPhrase) 
						nounPhraseBegin = False
					break

				word, tag = line.split()
				#print word, tag
				word, tag = word.strip().lower(), tag.strip()
				#print repr(tag)

				#check for adjective
				if tag == 'JJ' or tag == 'JJR' or tag == 'JJS':
					if nounPhraseBegin == True:
						writeFileIter.write(nounPhrase + ' ')
						nounPhraseBegin = False

					writeFileIter.write(word + ' ')
					continue
				#continue till we find nouns
				elif tag == 'NN' or tag == 'NNS' or tag == 'NNP' or tag == 'NNPS':
					if nounPhraseBegin == False:
						nounPhraseBegin = True
						nounPhrase = word
						continue

					nounPhrase += '_' + word
					continue
				#other tags
				else:
					if nounPhraseBegin == True:
						writeFileIter.write(nounPhrase + ' ')
						nounPhraseBegin = False

					continue

		except IOError as e:
			self.logger.info("IOError: %s: %s" % (e.filename, e.strerror))
		finally:
			if readFileIter is not None:
				readFileIter.close()
			if writeFileIter is not None:
				writeFileIter.close()
			self.logger.info("Get noun phrases end")


	def __extract_tags(self):
		self.logger.info("Extract tags begin")

		try:
			# open content file
			lemmatizedTextFileIter = open(self.lemmatizedTextFilePath, 'r')
			processedFileIter = open(self.processedFilePath, 'w+')

			for line in lemmatizedTextFileIter.readlines():
				tokenized = nltk.word_tokenize(line)
				tagged = nltk.pos_tag(tokenized)
				for word, tag in tagged:
					wordAndTag = word + ' ' + tag + '\n'
					processedFileIter.write(wordAndTag)
				#print tagged
				#time.sleep(2)

		except IOError as e:
			self.logger.info("IOError: %s: %s" % (e.filename, e.strerror))
		finally:
			if lemmatizedTextFileIter is not None:
				lemmatizedTextFileIter.close()
			if processedFileIter is not None:
				processedFileIter.close()
			self.logger.info("Extract tags end")


	def __generate_plain_text(self):
			self.logger.info("Generate plain text begin")

			try:
				pageoutFileIter = open(self.pageoutputFilePath, 'r')
				plaintextFileIter = open(self.plaintextFilePath, 'w+')

				iterparser = et.iterparse(pageoutFileIter, events=('start','end'))
				_, root = iterparser.next()

				for event, element in iterparser:
					if event == 'start':
						continue

					# Start event is required to get the root of the tree
					#print element.tag
					if element.tag == 'title':
						pagePlain = ''
						pagePlain = element.text + ' '

					if element.tag == 'text':
						pagePlain += element.text + '\n\n'

						#keep only alphanumerics, full stops and new line othrwise 
						#whole file will be read as a single line
						pagePlain = re.sub('[^a-zA-Z\.\n]+', ' ', pagePlain)
						#print pagePlain
						#print type(pagePlain)
						plaintextFileIter.write(pagePlain)
						root.clear()  #clear root to remove parsed part and reclaim memory

			except IOError as e:
				self.logger.info("IOError: %s: %s" % (e.filename, e.strerror))
			finally:
				if pageoutFileIter is not None:
					pageoutFileIter.close()
				if plaintextFileIter is not None:
					plaintextFileIter.close()
				self.logger.info("Generate plain text end")

	def __lemmatize_text(self):
		self.logger.info("Lemmatize text begin")

		try:
			readFileIter = open(self.plaintextFilePath, 'r')
			writeFileIter = open(self.lemmatizedTextFilePath, 'w+')
			
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



if __name__ == '__main__':
	preprocessor = Preprocessor()
	preprocessor.preprocess()
