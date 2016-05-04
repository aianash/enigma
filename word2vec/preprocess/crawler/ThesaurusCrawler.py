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

from nltk.corpus import wordnet
from bs4 import BeautifulSoup

#check for python version
python2 = False
if list(sys.version_info)[0] == 2:
	python2 = True

class ThesaurusCrawler:
	def __init__(self):
		program = os.path.basename(sys.argv[0])
		self.logger = logging.getLogger(program)
		self.thesaurusAdjectiveFilePath = "../../datafiles/preprocesseddata/thesaurusadjectives"

		#initialize logger
		logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
		logging.root.setLevel(level=logging.INFO)
		logging.getLogger("requests").setLevel(logging.WARNING)
		self.logger.info("running %s" % ' '.join(sys.argv))

		
	def crawl_thesaurus(self):
		self.logger.info("Crawl thesaurus begin")

		try:
			writeFileIter = open(self.thesaurusAdjectiveFilePath, 'w+')

			word_list_seed = self.get_wordnet_adjectives()
			word_list_seed = sorted(word_list_seed)
			
			progress = 0
			word_list_len = len(word_list_seed)

			self.logger.info("Words to crawl : {0}".format(word_list_len))
			for word in word_list_seed:
				word_context_tuple = self.get_word_context(word)
				json_word_context = self.get_json_word_context(word_context_tuple)
				writeFileIter.write(json_word_context + '\n')
				
				progress = progress + 1
				if progress % 1000 == 0:
					self.logger.info("Processed {0} words".format(progress))

			writeFileIter.close()				
		except IOError as e:
			self.logger.info("IOError: %s: %s" % (e.filename, e.strerror))
			raise e
		except Exception as e:
			self.logger.info("Exception occured while crawling thesaurus")
			raise e
		finally:
			if writeFileIter is not None:
				writeFileIter.close()
			self.logger.info("Crawl thesaurus end")


	def get_page(self, url):
		if url is None:
			self.logger.info("Error: Invalid url")
			return
		try:
			data = BeautifulSoup(requests.get(url).text)
			return data
		except Exception as e:
			self.logger.info("No Content in Thesaurus for url {0}".format(url))
			return None
			#raise e


	def get_synonyms(self, data, formatted = False):
		if data is None:
			self.logger.info("Error: data value in None")
		else:
			try:
				terms = data.select("div#filters-0")[0].findAll("li")
				li = []
				for t in terms:
					#print t
					li.append(t.select("span.text")[0].getText())
				if formatted:
					return {term: li}
				return li
			except Exception as e:
				self.logger.info("No Synonyms in this data")
				return []


	def get_related_words(self, data):
		if data is None:
			self.logger.info("Error: data value in None")
			return	
		try:
			content = data.select("div#content")[0]
			#print content
			#related word list class is box syn_of_syns oneClick-area
			related_class = content.findAll("div", {"class" : "box syn_of_syns oneClick-area"})
			#print related_class

			related_word_list = []
			for div in related_class:
			 	word = div.find('a').getText()
				terms = div.findAll("li")
				word_list = []
				for term in terms:
					word_list.append(term.getText())
				related_word_list.append((word, word_list))
			return related_word_list
			
		except Exception as e:
			self.logger.info("No related words in this data")
			return []


	def get_num_pages(self, data):
		if data is None:
			self.logger.info("Error: data value in None")
			return
		try:
			content = data.select("div#content")[0]
			#print content
			#see number of pages in paginator div
			paginator_class = content.find("div", {"id" : "paginator"})
			if paginator_class is None:
				num_pages = 1
			else:	
				page_divs = paginator_class.findAll("div")
				for div in page_divs:
					num_pages = div.find('a').getText()

			return int(num_pages)	

		except Exception as e:
			self.logger.info("Unable to find number of pages in data")
			raise e


	def get_word_context(self, word):
		if len(word.split()) > 1:
			self.logger.info("Error: A term must be only a single word")
			return

		try:
			url = "http://www.thesaurus.com/browse/{0}".format(word)
			data = self.get_page(url)
			#some adjectives from wordnet may not be present in thesaurus
			if data is None:
				return (word, [], [])
			
			num_pages = self.get_num_pages(data)

			syn_list = []
			related_word_list = []
			for i in range(num_pages):
				page_url = "http://www.thesaurus.com/browse/{0}/{1}".format(word, i + 1)
				page_data = self.get_page(page_url)
				if i == 0:
					syn_list = self.get_synonyms(page_data, formatted=False)
				related_word_list.extend(self.get_related_words(data))

			return (word, syn_list, related_word_list)
		except Exception as e:
			self.logger.info("Unable to find whole word context for word {0}".format(word))
			raise e


	def get_json_word_context(self, word_sys_related_tuple):
		if word_sys_related_tuple is None:
			self.logger.info("Error: Invalid word context tuple")
			return

		try:
			word, syn_list, related_word_list = word_sys_related_tuple
			related_words_map = {}
			for rel_word, rel_word_list in related_word_list:
				related_words_map[rel_word] = rel_word_list

			context_map = {}
			context_map["synonyms"] = syn_list
			context_map["related_words"] = related_words_map

			obj_map = {}
			obj_map["word"] = word
			obj_map["context"] = context_map

			return json.dumps(obj_map, encoding='utf-8')

		except Exception as e:
			self.logger.info("Unable to cretae json for word context")
			raise e


	def print_json_context(self, json_word_context):
		if json_word_context is None:
			self.logger.info("Error: Invalid word context tuple")
			return

		try:
			python_obj = json.loads(json_word_context)
			print json.dumps(python_obj, indent=4)

		except Exception as e:
			self.logger.info("Unable to print json for word context")
			raise e

	def get_wordnet_adjectives(self):
		try:
			#get all adjective synsets
			all_synset_list = []
			for synset in wordnet.all_synsets('a'):
				all_synset_list.append(synset)
			#print len(all_synset_list)			

			word_list_seed = set()
			for valueset in all_synset_list:
				word = valueset.name().partition('.')[0]
				word_list_seed.add(word)
				
			#print len(word_list_seed)
			return list(word_list_seed)			
		
		except Exception as e:
			self.logger.info("Unable to get adjective list from wordnet")
			raise e

	def execute(self):
		try:
			self.crawl_thesaurus()
		except Exception as e:
			exc_type, exc_value, exc_traceback = sys.exc_info()
			print exc_type
			print exc_value
			print(repr(traceback.format_tb(exc_traceback)))
			print("LINE WHERE EXCEPTION OCCURED : ", exc_traceback.tb_lineno)


if __name__ == '__main__':
	thesaurus_crawler = ThesaurusCrawler()
	thesaurus_crawler.execute()
