#!/usr/env python

import logging
import os.path
import bz2
import sys
import time
import traceback
import json

from nltk.corpus import wordnet


class WordnetUtil:
	def __init__(self):
		program = os.path.basename(sys.argv[0])
		self.logger = logging.getLogger(program)
		self.wordnetAdjFilePath = "../datafiles/preprocesseddata/wordnetadjectives"

		#initialize logger
		logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
		logging.root.setLevel(level=logging.INFO)
		self.logger.info("running %s" % ' '.join(sys.argv))


	def __get_syn_values(self, word):
		if word is None:
			self.logger.info("Error: Invalid word")
			return
		try:
			words_synsets = wordnet.synsets(word, 'a')
			#print words_synsets
			word_syn_list = []
			for synset in words_synsets:
				for lemma in wordnet.synset(synset.name()).lemmas():
					#print lemma.name()
					word_syn_list.append(lemma.name())
			return word_syn_list
		except Exception as e:
			self.logger.info("Exception occured while getting lemma names for word {0}".format(word))
			raise e
		
	def generate_wordnet_synsets(self):
		self.logger.info("Generate wordnet synsets begin")

		try:
			word_seed_list = set()
			#get all adjective synsets
			for synset in wordnet.all_synsets('a'):
				word = synset.name().partition('.')[0].strip()
				word_seed_list.add(word)
			#print len(all_synset_list)			

			word_syn_map = {}
			
			for word in word_seed_list:
				#print word
				word_syn_list = self.__get_syn_values(word)
				if word_syn_list is not None:
					word_syn_list = [w for w in word_syn_list if w != word]
				else:
					word_syn_list = []

				#print word_list_synset	
				if word not in word_syn_map:
					word_syn_map[word] = word_syn_list
				else:
					word_syn_map[word].extend(word_syn_list)

			#print len(word_list_seed)
			writeFileIter = open(self.wordnetAdjFilePath, 'w+')
			#print word_list_synset_map['able']
			for word, syn_list in sorted(word_syn_map.items()):
				json_word_context = self.get_json_word_context(word, syn_list)
			 	writeFileIter.write(json_word_context + '\n')
			writeFileIter.close()	

		except IOError as e:
			self.logger.info("IOError: %s: %s" % (e.filename, e.strerror))
			raise e
		except Exception as e:
			self.logger.info("Exception occured while generating synsets")
			raise e

		finally:
			if writeFileIter is not None:
				writeFileIter.close()
			self.logger.info("Generate wordnet synsets end")

	def get_json_word_context(self, word, syn_list):
		if syn_list is None or word is None:
			self.logger.info("Error: Invalid word context list")
			return

		try:
			context_map = {}
			context_map["synonyms"] = syn_list

			obj_map = {}
			obj_map["word"] = word
			obj_map["context"] = context_map

			return json.dumps(obj_map, encoding='utf-8')

		except Exception as e:
			self.logger.info("Unable to cretae json for word context")
			raise e


	def execute(self):
		try:
			#Generate maps
			self.generate_wordnet_synsets()
		except Exception as e:
			exc_type, exc_value, exc_traceback = sys.exc_info()
			print exc_type
			print exc_value
			print(repr(traceback.format_tb(exc_traceback)))
			print("LINE WHERE EXCEPTION OCCURED : ", exc_traceback.tb_lineno)


if __name__ == '__main__':
	word_net_util = WordnetUtil()
	word_net_util.execute()
