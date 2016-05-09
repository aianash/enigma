#!/usr/env python

import time
import os
import sys
import logging
import re
import networkx as nx #using pagerank implementation from here
import traceback

try:
	import cpickle as pickle
except:
	import pickle

try:
	import xml.etree.cElementTree as et
except:
	import xml.etree.ElementTree as et


class Pagerank:

	def __init__(self):
		program = os.path.basename(sys.argv[0])
		self.logger = logging.getLogger(program)
		self.pageoutputFilePath = "../datafiles/preprocesseddata/pageoutput.xml"
		self.nodeLabelFilePath = "../datafiles/pagerankdata/linkslabel.pkl"
		self.labelToNodeFilePath = "../datafiles/pagerankdata/labeltolinks.pkl"
		self.nodeGraphFilePath = "../datafiles/pagerankdata/linksdigraph.pkl"
		self.newSeedsFilePath = "../datafiles/seeddata/newseeds"
		self.seedFilePath = "../datafiles/seeddata/seeds"
		self.seedToLinkPair = "../datafiles/pagerankdata/seedToLinkPair"

		# initialize logger
		logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
		logging.root.setLevel(level=logging.INFO)
		self.logger.info("running %s" % ' '.join(sys.argv))

	#page rank algorithm
	def getPagerank(graph, damping=0.85, epsilon=1.0e-8):
		inlink_map = {}
		outlink_counts = {}

		def new_node(self, node):
			if node not in inlink_map: inlink_map[node] = set()
			if node not in outlink_counts: outlink_counts[node] = 0e-8

		for tail_node, head_node in graph:
			new_node(tail_node)
			new_node(head_node)
			if tail_node == head_node: continue

			if tail_node not in inlink_map[head_node]:
				inlink_map[head_node].add(tail_node)
				outlink_counts[tail_node] += 1

		all_nodes = set(inlink_map.keys())
		for node, outlink_count in outlink_counts.items():
			if outlink_count == 0:
				outlink_counts[node] = len(all_nodes)
				for l_node in all_nodes: inlink_map[l_node].add(node)

		initial_value = 1 / len(all_nodes)
		ranks = {}
		for node in inlink_map.keys(): ranks[node] = initial_value

		new_ranks = {}
		delta = 1.0
		n_iterations = 0
		while delta > epsilon:
			new_ranks = {}
			for node, inlinks in inlink_map.items():
				new_ranks[node] = ((1 - damping) / len(all_nodes)) + (damping * sum(ranks[inlink] / outlink_counts[inlink] for inlink in inlinks))
			delta = sum(abs(new_ranks[node] - ranks[node]) for node in new_ranks.keys())
			ranks, new_ranks = new_ranks, ranks
			n_iterations += 1

		return ranks, n_iterations


	def getPagerankNetworkx(self, linkList, numNewSeeds):
		self.logger.info("Get page ranks and get new seeds begin")
		idToLinkMap = {}
		newSeeds = []
		edgeList = []

		try:
			#create edgelist
			for (sourceLink, destLink) in linkList:
				edgeList.append((int(sourceLink), int(destLink)))
			#cretae directed graph
			diGraph = nx.from_edgelist(edgeList, create_using=nx.DiGraph())
			#evaluate pageranks
			linkPageRanks = nx.pagerank(diGraph)
			#lode node to id map
			idToLinkMap = pickle.load(open(self.labelToNodeFilePath, 'rb'))
			#sort page rank in descending order and get first thousand for new seeds

			for linkId, pagerank in sorted(linkPageRanks.items(), key = lambda x:x[1], reverse=True):
				if numNewSeeds == 0:
					break
				newSeeds.append((idToLinkMap[linkId], pagerank))
				numNewSeeds = numNewSeeds - 1

			#test geopy
			text = ''
			for link in newSeeds:
				text = text + ' ' + link[0]
			import geograpy
			places = geograpy.get_place_context(text=text)
			print places.countries

			#write new seeds to a file
			newSeedsFileIter = open(self.newSeedsFilePath, 'w+', 1048576) #1MB buffer
			for link in newSeeds:
				newSeedsFileIter.write(link[0] + ' ' + str(link[1]) + '\n')
			newSeedsFileIter.close()

			#pr_max = max(pr.items(), key= lambda x: x[1])
			#print '\nElement with the highest PageRank: {0}'.format(pr_max[0]+1)

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
			if newSeedsFileIter is not None:
				newSeedsFileIter.close()
			self.logger.info("Generate page ranks and get new seeds end")


	def createNodeGraph(self):
		self.logger.info("Generate node graph begin")
		nodeId = 1
		linkToIdMap = {}
		idToLinkMap = {}
		sourceToDestLinkSet = set()

		# \[\[                 Opening [[
		#	   (               Capture grp 1
		#		  [^\[\]|]*    Optional chars, NOT open/close brackets, nor |
		#                      (this can be changed to required with a + quantifier)
		#	   )               End grp 1
		#	 [^\[\]]*          Optional chars, NOT open/close brackets
		# \]\]                 Closing ]]
		linkPattern = re.compile(r"\[\[([^\[\]|]*)[^\[\]]*\]\]")

		try:
			pageoutFileIter = open(self.pageoutputFilePath, 'r')
			#get seed titles and populate array, 
			seedLinks = [links.rstrip('\n') for links in open(self.seedFilePath)]
			seedLinks = [links.strip() for links in seedLinks]

			#parse wiki pages output to get links pair
			iterparser = et.iterparse(pageoutFileIter, events=('start','end'))
			_, root = iterparser.next()

			for event, element in iterparser:
				if event == 'start':
					continue

				# Start event is required to get the root of the tree
				#print element.tag
				if element.tag == 'title':
					node = element.text.strip()
					linkToIdMap[node] = nodeId
					idToLinkMap[nodeId] = node
					nodeId = nodeId + 1

				if element.tag == 'text':
					nodeText = element.text.strip()
					#find alll links in the text
					links = linkPattern.findall(nodeText)
					#add links pair into map
					links = [link.strip() for link in links]

					if node in seedLinks:
						for link in links:
							sourceToDestLinkSet.add((node, link))
					else:
						for link in links:
							if link in seedLinks:
								sourceToDestLinkSet.add((node, link))
					root.clear()  #clear root to remove parsed part and reclaim memory

			pageoutFileIter.close()
			#write graph and node map to a file
			nodeGraphFileIter = open(self.nodeGraphFilePath, 'wb+')
			pickle.dump(list(sourceToDestLinkSet), nodeGraphFileIter)
			nodeGraphFileIter.close()

			nodeLabelFileIter = open(self.nodeLabelFilePath, 'wb+')
			pickle.dump(linkToIdMap, nodeLabelFileIter)
			nodeLabelFileIter.close()

			labelToNodeFileIter = open(self.labelToNodeFilePath, 'wb+')
			pickle.dump(idToLinkMap, labelToNodeFileIter)
			labelToNodeFileIter.close()

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
			if pageoutFileIter is not None:
				pageoutFileIter.close()
			if nodeGraphFileIter is not None:
				nodeGraphFileIter.close()
			if nodeLabelFileIter is not None:
				nodeLabelFileIter.close()
			if labelToNodeFileIter is not None:
				labelToNodeFileIter.close()
			self.logger.info("Generate node graph end")


	def loadNodeGraphAndCreateIdGraph(self):
		self.logger.info("Load node graph begin")
		linkToIdMap = {}
		sourceToDestLinkList = []
		sourceToDestLinkIdList = []

		try:
			linkToIdMap = pickle.load(open(self.nodeLabelFilePath, 'rb'))
			sourceToDestLinkList = pickle.load(open(self.nodeGraphFilePath, 'rb'))
			seedToLinkPair = open(self.seedToLinkPair, 'w+')

			#convert graph from title strings into node ids
			for (sourceLink, destLink) in sourceToDestLinkList:
				if destLink not in linkToIdMap:
					continue
				seedToLinkPair.write(sourceLink + ' ' + destLink + '\n')
				sourceToDestLinkIdList.append((linkToIdMap[sourceLink], linkToIdMap[destLink]))

			linkToIdMap.clear()
			sourceToDestLinkList[:]

			self.logger.info("Load node graph end")
			return sourceToDestLinkIdList

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
			if seedToLinkPair is not None:
				seedToLinkPair.close()


if __name__ == '__main__':
	pagerank = Pagerank()
	#pagerank.createNodeGraph()
	sourceToDestLinkIdMap = pagerank.loadNodeGraphAndCreateIdGraph()
	pagerank.getPagerankNetworkx(sourceToDestLinkIdMap, 9000)

