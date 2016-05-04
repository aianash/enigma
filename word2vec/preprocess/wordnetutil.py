#!/usr/env python

import logging
import os.path
import bz2
import sys
import time
import traceback

from nltk.corpus import wordnet

from utils import PyDictionary
from PyDictionary import PyDictionary


class WordnetUtil:
	def __init__(self):
		program = os.path.basename(sys.argv[0])
		self.logger = logging.getLogger(program)
		self.wordnetAdjFilePath = "../datafiles/preprocesseddata/wordnetadj"

		#initialize logger
		logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
		logging.root.setLevel(level=logging.INFO)
		self.logger.info("running %s" % ' '.join(sys.argv))


	def __get_synset_values(self, synset):
		syn_word_list = []
		for lemma in wordnet.synset(synset.name()).lemmas():
			syn_word_list.append(lemma.name())
		return syn_word_list

	def get_similarto_words(self):
		synsets = wordnet.synsets("modern", 'a')
		for synset in synsets:
			similarto = wordnet.synset(synset.name()).lemmas()
			print similarto[0].derivationally_related_forms()
		
	def generate_wordnet_synsets(self):
		self.logger.info("Generate wordnet synsets begin")

		try:
			#get all adjective synsets
			all_synset_list = []
			for synset in wordnet.all_synsets('a'):
				all_synset_list.append(synset)
			print len(all_synset_list)			

			word_list_synset_map = {}
			word_list_pydict_map = {}
			word_list_seed = []

			for valueset in all_synset_list:
				word = valueset.name().partition('.')[0]
				word_list_seed.append(word)
				#print word
				word_list_synset = self.__get_synset_values(valueset)
				if word_list_synset is not None:
					word_list_synset = [w for w in word_list_synset if w != word]
				else:
					word_list_synset = []

				word_list_synset_map[word] = word_list_synset
				#filter(lambda a: a != word, word_list_synset)
				#print word_list_synset

			py_dict = PyDictionary(word_list_seed)
			word_list_pydict = py_dict.getSynonyms(False)
			print type(word_list_pydict)

			#filter(lambda a: a != word, word_list_pydict)
			if word_list_pydict is not None:
				word_list_pydict = [w for w in word_list_pydict if w != word]
			else:
				word_list_pydict = []
			#print word_list_pydict

				syn_word_list.extend(word_list_synset)
				syn_word_list.extend(word_list_pydict)
				syn_word_list.append(word)
				syn_word_list.extend(word_list_pydict)
				syn_word_list.extend(word_list_synset)

				#print syn_word_list
				#break

			writeFileIter = open(self.wordnetAdjFilePath, 'wb+')

			print len(syn_word_list)
			for word in syn_word_list:
			 	writeFileIter.write(word + ' ')

			writeFileIter.close()				
		except IOError as e:
			self.logger.info("IOError: %s: %s" % (e.filename, e.strerror))
		except Exception as e:
			exc_type, exc_value, exc_traceback = sys.exc_info()
			print exc_type
			print exc_value
			print(repr(traceback.format_tb(exc_traceback)))
			print("LINE WHERE EXCEPTION OCCURED : ", exc_traceback.tb_lineno)

		finally:
			if writeFileIter is not None:
				writeFileIter.close()
			self.logger.info("Generate wordnet synsets end")

	def execute(self):
		try:
			#Generate maps
			#word_net_util.generate_wordnet_synsets()
			#word_net_util.get_similarto_words()
			self.getPage()
		except Exception as e:
			exc_type, exc_value, exc_traceback = sys.exc_info()
			print exc_type
			print exc_value
			print(repr(traceback.format_tb(exc_traceback)))
			print("LINE WHERE EXCEPTION OCCURED : ", exc_traceback.tb_lineno)


if __name__ == '__main__':
	word_net_util = WordnetUtil()
	word_net_util.execute()
