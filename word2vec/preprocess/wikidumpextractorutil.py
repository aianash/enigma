#!/usr/env python

import logging
import os.path
import bz2
import sys
import re
import time
import traceback
import operator

try:
	import xml.etree.cElementTree as et
except ImportError:
	import xml.etree.ElementTree as et

try:
	import cpickle as pickle
except:
	import pickle



class WikiExtUtil:
	def __init__(self):
		program = os.path.basename(sys.argv[0])
		self.logger = logging.getLogger(program)
		self.dumpFilePath = "../wikidump/enwiki-latest-pages-articles.xml.bz2"
		self.seedFilePath = "../datafiles/seeddata/seeds"
		self.newseedFilePath = "../datafiles/seeddata/newseeds"
		self.pageRankedSeedFilePath = "../datafiles/seeddata/prseeds"
		self.titleListFilePath = "../datafiles/wikihelpermaps/title-list.pkl"
		self.titleToPageMapFilePath = "../datafiles/wikihelpermaps/title-to-page-map.pkl"
		self.titleToRedirectTitleMapFilePath = "../datafiles/wikihelpermaps/title-to-redirect-title-map.pkl"
		self.depthLinksFilePath = "../datafiles/seeddata/depth1links"
		self.pageRankedDepthLinksFilePath = "../datafiles/seeddata/prdepth1links"
		self.parseOutputFilePath = "../datafiles/preprocesseddata/pageoutput.xml"
		self.nameSpace = '{http://www.mediawiki.org/xml/export-0.10/}'
		self.totalPages = 17000000 #number of pages in current wikimpedia dump(8th march)


		#initialize logger
		logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
		logging.root.setLevel(level=logging.INFO)
		self.logger.info("running %s" % ' '.join(sys.argv))



	def extract_seed_links(self, seedFilePath):
		self.logger.info("Extract seed links begin")

		try:
			#open output file, creates if none is there. For seedlinks rewrite the file
			outfile = open(self.parseOutputFilePath, 'w+', 1048576) #1MB buffer

			#add root element 
			outfile.write("<root>\n")

			requiredLinksMap = self.__filter_links(seedFilePath)

			self.__parse_wiki_dump(outfile, requiredLinksMap)

		except IOError as e:
			self.logger.info("IOError: %s: %s" % (e.filename, e.strerror))
			raise e
		finally:
			if outfile is not None:
				outfile.close()
			self.logger.info("Extract seed links end")


	def get_depth1_links(self, seedFilePath, depthLinksFilePath):
		self.logger.info("Get depthlinks begin")
		# \[\[                 Opening [[
		#	   (               Capture grp 1
		#		  [^\[\]|]*    Optional chars, NOT open/close brackets, nor |
		#                      (this can be changed to required with a + quantifier)
		#	   )               End grp 1
		#	 [^\[\]]*          Optional chars, NOT open/close brackets
		# \]\]                 Closing ]]
		linkPattern = re.compile(r"\[\[([^\[\]|]*)[^\[\]]*\]\]")

		try:
			#get seed titles and populate array, 
			seedLinks = [links.rstrip('\n') for links in open(seedFilePath)]
			seedLinks = [links.strip() for links in seedLinks]
			#print seedLinks
			
			depth1Links = []
			depthLinksFiltered = set()
			with open(self.parseOutputFilePath, 'r') as infile: #autoclose the file 
				for line in infile:
					links = linkPattern.findall(line)
					depth1Links.extend(links)

			#print depth1Links
			#remove duplicate links and links from seed
			depth1Links = [links.strip() for links in depth1Links]
			#depth1Links = [links.decode('utf-8') for links in depth1Links]
			depth1Links = set(depth1Links) - set(seedLinks) #depth1Links is a set here
			#print depth1Links

			#filter links that are not page titles(can be file links etc.)
			self.logger.info("Loading title list")
			titleList = pickle.load(open(self.titleListFilePath, 'rb'))
			self.logger.info("Title list loaded")

			#filter titles that are redirection to another article(there may be multiple links pointing 
			#to same article page)
			self.logger.info("Loading title to redirection title map")
			titleToRedirectionTitleMap = pickle.load(open(self.titleToRedirectTitleMapFilePath, 'rb'))
			self.logger.info("Title to redirection title map loaded")

			#print type(depth1Links)
			linksFound = 0
			for link in depth1Links:
				linksFound = linksFound + 1
				if (linksFound % 500 == 0):
				 	self.logger.info("Found %d links" % (linksFound))
				#print i, link
				#print "link:",link
				#print type(link)
				#print titleList[link]
				#print type(titleList[link])
				link = link.decode(encoding='utf-8')
				#print type(link)
				if link in titleList:
					if link in titleToRedirectionTitleMap:	
						sourceLink = titleToRedirectionTitleMap[link]
						#print "sourcelink:", sourceLink
						if sourceLink is not None:
							sourceLink = sourceLink.encode(encoding='utf-8')
							depthLinksFiltered.add(sourceLink) #depth1Links is a set so link will be added only once
					else:
						link = link.encode(encoding='utf-8')
						depthLinksFiltered.add(link)
				
			#clear memory as soon as it is not required		
			del titleList[:] 
			titleToRedirectionTitleMap.clear()

			#links same as seed links may have been included due to redirection
			#so remove them
			depthLinksFiltered = depthLinksFiltered - set(seedLinks)
			
			#depthLinksFiltered = depth1Links
			#open output file, creates if none is there
			depthLinksFileIter = open(depthLinksFilePath, 'w+', 1048576) #1MB buffer
			for link in list(depthLinksFiltered):
				depthLinksFileIter.write(link + '\n')
				
		except IOError as e:
			self.logger.info("IOError: %s: %s" % (e.filename, e.strerror))
		except Exception as e:
			#self.logger.info("Unexpected error, %s " % (sys.exc_info()[0]))
			exc_type, exc_value, exc_traceback = sys.exc_info()
			print exc_type
			print exc_value
			print(repr(traceback.format_tb(exc_traceback)))
			print("LINE WHERE EXCEPTION OCCURED : ", exc_traceback.tb_lineno)
			raise e
		finally:
			if depthLinksFileIter is not None:
				depthLinksFileIter.close()
			self.logger.info("Get depthlinks end")

		
	def extract_depth1_links(self, depthLinksFilePath):
		self.logger.info("Extract depth1 links begin")

		try:
			#open output file, creates if none is there. For seedlinks rewrite the file
			outfile = open(self.parseOutputFilePath, 'a+', 1048576) #1MB buffer
			outfile.write("\n")

			requiredLinksMap = self.__filter_links(depthLinksFilePath)

			self.__parse_wiki_dump(outfile, requiredLinksMap)

			#add end root element
			outfile.write("</root>")

		except IOError as e:
			self.logger.info("IOError: %s: %s" % (e.filename, e.strerror))
		finally:
			if outfile is not None:
				outfile.close()
			self.logger.info("Extract depth1 links end")


	def __filter_links(self, linksFilePath):
		try:
			self.logger.info("Filter links begin")
			#get depth1 links and populate array, 
			requiredLinks = [links.rstrip('\n') for links in open(linksFilePath)]
			requiredLinks = [links.strip() for links in requiredLinks]
			#print depth1Links
			print len(requiredLinks)
			#get required page offsets through title maps
			self.logger.info("Loading title to page index map")
			titleToPageMap = pickle.load(open(self.titleToPageMapFilePath, 'rb'))
			self.logger.info("Title to page index map loaded")
			requiredLinksMap = {}

			for link in requiredLinks:
				if link in titleToPageMap and titleToPageMap[link] is not None:
					requiredLinksMap[link] = titleToPageMap[link]
				else:
					self.logger.info("Link not found: %s" % link)

			titleToPageMap.clear() #It is a big map clear as soon as it is not required to reclaim memory
			self.logger.info("Filter links end")
			return requiredLinksMap
		except Exception as e:
			exc_type, exc_value, exc_traceback = sys.exc_info()
			print exc_type
			print exc_value
			print(repr(traceback.format_tb(exc_traceback)))
			print("LINE WHERE EXCEPTION OCCURED : ", exc_traceback.tb_lineno)
			raise e



	def __parse_wiki_dump(self, fileToWrite, linksDict):
		self.logger.info("Parse wiki dump begin")

		try:		
			with bz2.BZ2File(self.dumpFilePath) as dumpFileIter:
				# iterparser = et.iterparse(wikidump, events=('start','end'))
				# _, root = iterparser.next()
				# for event, element in iterparser:
				# 	if event == 'start':
				# 		continue

				# 	#process only end events with namespace, start event is required to get the root of the tree
				# 	#print element.tag
				# 	if element.tag == '{http://www.mediawiki.org/xml/export-0.10/}page':
				# 		print element.tag
				# 		numPagesParsed += 1
				# 		page = et.tostring(element)
				# 		pagexml = et.fromstring(page)
				# 		for child in pagexml:
				# 			if child.tag == '{http://www.mediawiki.org/xml/export-0.10/}title':
				# 				if child.text is not None and child.text.lower() in linksList:
				# 					numRelPages -= 1
				# 					fileToWrite.write(page)
				# 					#print page
				# 		root.clear() #clear root to remove parsed part and reclaim memory

				# 	if numRelPages == 0:
				# 		break
				numPagesParsed = 0
				numRelPages = len(linksDict)

				self.logger.info("Pages to parse %d" % (numRelPages))

				#sort linksDict based on page start index for fast parsing. 
				#Values are tuples (startindex, endindex)

				#print linksDict
				#for k,v in sorted(linksDict.items(), key = lambda x:x[1]):
				#	print k,v

				#sort and make a list of page indexes
				# sortedPageIndexes = sorted(linksDict.values(), key = lambda x:x[0])
				# print sortedPageIndexes
				# print type(sortedPageIndexes)

				for _, pageIndex in sorted(linksDict.items(), key = lambda x:x[1]):
					dumpFileIter.seek(pageIndex[0])
					fileToWrite.write(dumpFileIter.read(pageIndex[1] - pageIndex[0]))
					numPagesParsed += 1

				 	if (numPagesParsed % 1000 == 0):
				 		self.logger.info("Parsed %d pages" % (numPagesParsed))					

			self.logger.info("Parse wiki dump complete")

		except IOError as e:
			self.logger.info("IOError: %s: %s" % (e.filename, e.strerror))


	def get_pages_for_given_seeds(self, seedFilePath, outFilePath):
		self.logger.info("Get pages for seeds begin")

		try:
			#open output file, creates if none is there. For seedlinks rewrite the file
			outfile = open(outFilePath, 'w+', 1048576) #1MB buffer

			#add root element 
			outfile.write("<root>\n")

			requiredLinksMap = self.__filter_links(seedFilePath)

			self.__parse_wiki_dump(outfile, requiredLinksMap)

			#add end root element
			outfile.write("\n</root>")

		except IOError as e:
			self.logger.info("IOError: %s: %s" % (e.filename, e.strerror))
			raise e
		finally:
			if outfile is not None:
				outfile.close()
			self.logger.info("Extract seed links end")


if __name__ == '__main__':
	wikiExtUtil = WikiExtUtil()

	#parse dump
	# wikiExtUtil.extract_seed_links(wikiExtUtil.seedFilePath)
	# time.sleep(10)
	# wikiExtUtil.get_depth1_links(wikiExtUtil.seedFilePath, 
	#  	wikiExtUtil.depthLinksFilePath)
	# time.sleep(10) 
	# wikiExtUtil.extract_depth1_links(wikiExtUtil.depthLinksFilePath)
	wikiExtUtil.get_pages_for_given_seeds(wikiExtUtil.newseedFilePath, 
		wikiExtUtil.parseOutputFilePath)

	#perform page rank
	
