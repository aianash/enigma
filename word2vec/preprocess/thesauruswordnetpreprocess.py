#!/usr/env python

import logging
import os.path
import bz2
import sys
import time
import traceback
import requests
import re
import json


#check for python version
python2 = False
if list(sys.version_info)[0] == 2:
	python2 = True

class ThesaurusWordnetPreprocess:
	def __init__(self):
		program = os.path.basename(sys.argv[0])
		self.logger = logging.getLogger(program)
		self.thesaurusAdjectiveFilePath = "../datafiles/preprocesseddata/thesaurusadjectives"
		self.wordnetAdjectiveFilePath = "../datafiles/preprocesseddata/wordnetadjectives"
		self.adjectivePhraseFilePath = "../datafiles/preprocesseddata/adjectivephrases"

		#initialize logger
		logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
		logging.root.setLevel(level=logging.INFO)
		logging.getLogger("requests").setLevel(logging.WARNING)
		self.logger.info("running %s" % ' '.join(sys.argv))

		
	def preprocess(self):
		self.logger.info("Adjective preprocessing begin")

		try:
			readFileIter1 = open(self.thesaurusAdjectiveFilePath, 'r')
			readFileIter2 = open(self.wordnetAdjectiveFilePath, 'r')

			word_adj_map = self.get_adj_context(readFileIter1)
			word_adj_map_wordnet = self.get_adj_context(readFileIter2)
			
			readFileIter1.close()
			readFileIter2.close()
			#merge maps
			#print("read files")
			for word, syn_list in word_adj_map_wordnet.items():
				word_adj_map[word].extend(syn_list)

			#print("cretaed single map")
			writeFileIter = open(self.adjectivePhraseFilePath, 'w+')
			#create contexts file
			for word, syn_list in word_adj_map.items():
				if len(syn_list) == 0:
					print "No sysn found for word {0}".format(word)
					continue
				num_rem_words = len(syn_list) % 10
				num_rem_words = 10 - num_rem_words

				#synonyms from thesaurus are in the start which are more relavant
				i = 0
				j = 0
				while i < num_rem_words:
					if j == len(syn_list):
						j = 0
					syn_list.append(syn_list[j])
					j = j + 1
					i = i + 1

				#print len(syn_list)

				#break words in chunk of 10 
				i = 0
				while i  < len(syn_list):
					word_list = syn_list[i : i + 10]
					#print (word_list)
					file_line = word.encode("utf-8")
					#if word is not single word then combine it with underscore
					for syn_word in word_list:
						syn_word = '_'.join(syn_word.split(' '))
						file_line = file_line + ' ' + syn_word.encode("utf-8")
					#print file_line
					writeFileIter.write(file_line + '\n')
					i = i + 10
			
			writeFileIter.close()				
		except IOError as e:
			self.logger.info("IOError: %s: %s" % (e.filename, e.strerror))
			raise e
		except Exception as e:
			exc_type, exc_value, exc_traceback = sys.exc_info()
			print exc_type
			print exc_value
			print(repr(traceback.format_tb(exc_traceback)))
			print("LINE WHERE EXCEPTION OCCURED : ", exc_traceback.tb_lineno)
			self.logger.info("Exception occured while preprocessing")
			raise e
		finally:
			if readFileIter1 is not None:
				readFileIter1.close()
			if readFileIter2 is not None:
				readFileIter2.close()
			if writeFileIter is not None:
				writeFileIter.close()
			self.logger.info("Adjective preprocessing end")


	def get_adj_context(self, readFileIter):
		if readFileIter is None:
			self.logger.info("Error: Invalid file")
			return
		try:
			word_context_map = {}
			for line in readFileIter.readlines():
				data = json.loads(line)
				word_syn_list = []
				related_word_list = []
			
				word = data["word"]
				#print word
				word_syn_list = data["context"]["synonyms"]
				#print word_syn_list
				if data["context"].get("related_words") is not None:
					related_word_list = data["context"]["related_words"]
					#print related_word_list
					for related_word in related_word_list:
						continue
						#word_syn_list.append(related_word)
						#word_syn_list.extend(data["context"]["related_words"][related_word])

				word_context_map[word] = word_syn_list

			return word_context_map

		except Exception as e:
			exc_type, exc_value, exc_traceback = sys.exc_info()
			print exc_type
			print exc_value
			print(repr(traceback.format_tb(exc_traceback)))
			print("LINE WHERE EXCEPTION OCCURED : ", exc_traceback.tb_lineno)
			self.logger.info("Failed to read json")
			raise e

	def execute(self):
		try:
			self.preprocess()
		except Exception as e:
			self.logger.info("Exception occurec in script execution. Stopping...")


if __name__ == '__main__':
	thesaurus_wordnet_preprocess = ThesaurusWordnetPreprocess()
	thesaurus_wordnet_preprocess.execute()
