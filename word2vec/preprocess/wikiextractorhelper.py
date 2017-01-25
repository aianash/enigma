#!/usr/env python

import logging
import os.path
import bz2
import sys
import time

try:
	import xml.etree.cElementTree as et
except ImportError:
	import xml.etree.ElementTree as et

try:
	import cpickle as pickle
except:
	import pickle



class WikiExtHelper:
	def __init__(self):
		program = os.path.basename(sys.argv[0])
		self.logger = logging.getLogger(program)
		self.dumpFilePath = "../wikidump/enwiki-latest-pages-articles.xml.bz2"
		self.titleListFilePath = "../datafiles/wikihelpermaps/title-list.pkl"
		self.titleToPageMapFilePath = "../datafiles/wikihelpermaps/title-to-page-map.pkl"
		self.titleToRedirectTitleMapFilePath = "../datafiles/wikihelpermaps/title-to-redirect-title-map.pkl"
		self.pageStartTag = '<page>'
		self.pageEndTag = '</page>'		                           
		self.totalPages = 17000000 #number of pages in current wikimpedia dump(6th march)
		self.redirectedPages = 7155407 #number of redirected pages (https://dumps.wikimedia.org/enwiki/latest/)


		#initialize logger
		logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
		logging.root.setLevel(level=logging.INFO)
		self.logger.info("running %s" % ' '.join(sys.argv))

		
	def generateHelperMapFiles(self):
		self.logger.info("Generate map files begin")

		try:
			titleToPageMap = {}
			titleToRedirectTitleMap = {}
			pageList = []
			pageFound = False
			numPagesParsed = 0

			dumpFileIter = bz2.BZ2File(self.dumpFilePath, 'r')

			while True:
				line = dumpFileIter.readline()
				#print line
				if not line: #end of file, break the loop
					break

				#check for page start
				if pageFound == False:
					if self.pageStartTag in line:
						startByte = dumpFileIter.tell() - len(line)
						#print startByte
						pageList.append(line)
						#print line
						pageFound = True
						continue
					else:
						continue

				#add lines till page end 
				pageList.append(line)
				#print line

				#check for page end 
				if self.pageEndTag in line:
					endByte = dumpFileIter.tell()
					#print endByte
					pageFound = False
					
					#get title and check for redirection
					page = ''.join(pageList)
					pagexml = et.fromstring(page)
					#print page
					#print len(page)
					for child in pagexml:
						#assuming that xml is welformed so order of tags will be same for pages 
						#title(always present)->redirect(not present for every page->revision(always present)
						if child.tag == 'title':
							if child.text is not None:
								title = child.text.strip()
								titleToPageMap[title] = (startByte, endByte)
							continue

						if child.tag == 'revision':
							break

						if child.tag == 'redirect':
							if child.attrib['title'] is not None and title is not None:
								redirectionTitle = child.attrib['title']
								titleToRedirectTitleMap[title] = redirectionTitle.strip()
					#clear page
					del pageList[:]
					numPagesParsed += 1
					if numPagesParsed % 1000000 == 999999:
						self.logger.info("Parsed %d pages. %d percent complete." % (numPagesParsed, ((numPagesParsed * 100)/self.totalPages)))
				#print titleToPageMap
				#print titleToRedirectTitleMap

			# for titleName, pageBytes in titleToPageMap.iteritems():
			# 	dumpFileIter.seek(pageBytes[0])
			# 	print dumpFileIter.read(pageBytes[1] - pageBytes[0])

			dumpFileIter.close()
			#end of while, write map and title list to file
			titleListFileIter = open(self.titleListFilePath, 'wb+')
			titleList = list(titleToPageMap.keys())
			pickle.dump(titleList, titleListFileIter)
			titleList[:]
			titleListFileIter.close()

			titleToPageMapFileIter = open(self.titleToPageMapFilePath, 'wb+')
			pickle.dump(titleToPageMap, titleToPageMapFileIter)
			titleToPageMap.clear()
			titleToPageMapFileIter.close()

			titleToRedirectTitleMapFileIter = open(self.titleToRedirectTitleMapFilePath, 'wb+')
			pickle.dump(titleToRedirectTitleMap, titleToRedirectTitleMapFileIter)
			titleToRedirectTitleMap.clear()
			titleToRedirectTitleMapFileIter.close()

			#print len(titleToPageMap)
			#print len(titleToRedirectTitleMap)

		except IOError as e:
			self.logger.info("IOError: %s: %s" % (e.filename, e.strerror))
		except Exception as e:
			exc_type, exc_value, exc_traceback = sys.exc_info()
			print exc_type
			print exc_value
			print(repr(traceback.format_tb(exc_traceback)))
			print("LINE WHERE EXCEPTION OCCURED : ", exc_traceback.tb_lineno)

		finally:
			if dumpFileIter is not None:
				dumpFileIter.close()
			if titleListFileIter is not None:
				titleListFileIter.close()
			if titleToPageMapFileIter is not None:
				titleToPageMapFileIter.close()
			if titleToRedirectTitleMapFileIter is not None:
				titleToRedirectTitleMapFileIter.close()
			self.logger.info("Generate map files end")


	def getPageContent(self):
		self.logger.info("Get page content begin")

		#fileIter = bz2.BZ2File(self.dumpFilePath, 'r')

		try:
			titleToPageMap = {}
			titleToRedirectTitleMap = {}

			dumpFileIter = open(self.dumpFilePath, 'r')
			titleToPageMap = pickle.load(open(self.titleToPageMapFilePath, 'rb'))
			titleToRedirectTitleMap = pickle.load(open(self.titleToRedirectTitleMapFilePath, 'rb'))

			#print titleToPageMap
			#print titleToRedirectTitleMap

			for titleName, pageBytes in titleToPageMap.iteritems():
				dumpFileIter.seek(pageBytes[0])
				print dumpFileIter.read(pageBytes[1] - pageBytes[0])

		except IOError as e:
			self.logger.info("IOError: %s: %s" % (e.filename, e.strerror))
		finally:
			if dumpFileIter is not None:
				dumpFileIter.close()
			self.logger.info("Get page content end")



if __name__ == '__main__':
	wikiExtHelper = WikiExtHelper()

	#Generate maps
	wikiExtHelper.generateHelperMapFiles()
	#wikiExtHelper.getPageContent()
