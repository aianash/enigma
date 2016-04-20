#!/usr/env python

import nltk
import re
import time
import os
import sys
import logging

try:
	import xml.etree.cElementTree as et
except ImportError:
	import xml.etree.ElementTree as et


class Preprocessor:

	def __init__(self):
		program = os.path.basename(sys.argv[0])
		self.logger = logging.getLogger(program)
		self.pageoutputFilePath = "../datafiles/preprocessesddata/pageoutput.xml"
		self.plaintextFilePath = "../datafiles/preprocesseddata/plaintextoutput.txt"
		self.processedFilePath = "../datafiles/preprocesseddata/taggedwords.txt"
		self.nounphrasesFilePath = "../datafiles/preprocesseddata/nounphrases"

		# initialize logger
		logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
		logging.root.setLevel(level=logging.INFO)
		self.logger.info("running %s" % ' '.join(sys.argv))

	def preprocess(self):
		self.logger.info("Preprocessing begin")

		try:
			# open content file
			#self.__generate_plain_text()
			#self.__extract_tags()
			self.__get_noun_phrases()

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


	def __extract_tags(self):
		self.logger.info("Extract tags begin")

		try:
			# open content file
			plaintextFileIter = open(self.plaintextFilePath, 'r')
			processedFileIter = open(self.processedFilePath, 'w+')

			for line in plaintextFileIter.readlines():
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
			if plaintextFileIter is not None:
				plaintextFileIter.close()
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


if __name__ == '__main__':
	preprocessor = Preprocessor()
	preprocessor.preprocess()
