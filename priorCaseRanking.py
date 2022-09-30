import os
import pandas as pd
import nltk
import string 
import re
import pickle
import json
import shutil
import spacy
import subprocess 
from spacy.lang.en import English
import itertools
from tqdm import tqdm
from sklearn import preprocessing
import numpy
import multiprocessing
from time import time 
from scipy import spatial
import torch
import swifter
import time
import random

from rank_bm25 import BM25Plus
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity



nlp = English()

sp = spacy.load("en_core_web_sm")

wn = nltk.WordNetLemmatizer()


class PriorCaseRank():

	def __init__(self, datasetName, stopwords='stopwords.txt', queryFile = "FIRE2017-IRLeD-track-data/Task_2/Current_Cases", documentFile = "FIRE2017-IRLeD-track-data/Task_2/Prior_Cases", resultFile='Results.txt', relevantFile = '', test=True):

		self.stopwords = stopwords
		self.stopwordLst = []
		if(datasetName=="FIRE2017"):
			self.queryFile = "FIRE2017-IRLeD-track-data/Task_2/Current_Cases" 
			self.documentFile = "FIRE2017-IRLeD-track-data/Task_2/Prior_Cases"
		elif(datasetName=="COLIEE2021"):
			self.test = test
			if(test):
				self.queryFile = "COLIEE2021/Query_Cases" 
				self.documentFile = "COLIEE2021/Noticed_Cases"
				self.qrel= "COLIEE2021/qrelCOLIEE2021.txt"
			else:
				self.queryFile = "COLIEE2021/Query_Cases_train" 
				self.documentFile = "COLIEE2021/Noticed_Cases_train"
				self.qrel= "COLIEE2021/fqrelCOLIEE2021_train.txt"
		else:
			print("ERROR dataset name not found")

		self.resultFile = resultFile
		self.queryDataFrame = None
		self.documentDataFrame = None
		self.querySentencesDataFrame = None
		self.documentSentencesDataframe = None
		self.w2v_model = None
		self.d2v_model = None
		self.bm25 = None
		self.customStopwordLst = [" " , "\n", "'s", "...", "wa", "\n ", " \n", " \n ", "\xa0", "date_suppresed"]
		self.suppressedCitationWords = ['CITATION_SUPPRESSED', 'REFERENCE_SUPPRESSED', 'FRAGMENT_SUPPRESSED', 'CITATION']
		self.romanNumerals = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII"]
		self.datasetName = datasetName

		if(test):
			self.pickleFilePath1_COLIEE2021="COLIEE2021/clean_query_coliee2021.pickle"
			self.pickleFilePath2_COLIEE2021="COLIEE2021/clean_documents_coliee2021.pickle"
		else:
			self.pickleFilePath1_COLIEE2021="COLIEE2021/train_pickles/clean_query_coliee2021.pickle"
			self.pickleFilePath2_COLIEE2021="COLIEE2021/train_pickles/clean_documents_coliee2021.pickle"
		self.pickleFilePath1_FIRE2017 = "FIRE2017-IRLeD-track-data/clean_query_fire2017.pickle"
		self.pickleFilePath2_FIRE2017 = "FIRE2017-IRLeD-track-data/clean_documents_fire2017.pickle"


	def createQueryCasesFolder_COLIEE2021(self):
		with open("COLIEE2021/test_labels.json", "r") as f:
			data = json.load(f)
			for currentCase in data.keys():
				shutil.copy("COLIEE2021/task1_test/redacted/"+currentCase, "COLIEE2021/Query_Cases/"+currentCase)

	def createNoticedCasesFolder_COLIEE2021(self):
		with open("COLIEE2021/test_labels.json", "r") as f:
			data = json.load(f)
			for currentCase in data.keys():
				for file in data[currentCase]:
					shutil.copy("COLIEE2021/task1_test/redacted/"+file, "COLIEE2021/Noticed_Cases/"+file)

	def createQrelFile_COLIEE2021(self):
		f = open("COLIEE2021/test_labels.json", "r")
		qrelFile = open(self.qrel, "w+")
		data = json.load(f)

		for query in data:
			docs = []
			for doc in data[query]:
				if(doc not in docs):
					docs.append(doc)
					qrelFile.write(query.replace(".txt","")+" 0 "+doc.replace(".txt","") + " 1 \n")

		f.close()
		qrelFile.close()

	
	def calculateRecall(self, results, labels, topn):
		lf = open(labels, "r")
		rf = open(results, "r")
		if(self.datasetName=="COLIEE2021"):
			results_lines = rf.readlines()
			labels_json = json.load(lf)
			rel_num = 0
			rel_cases = 0
			for label in labels_json.keys():
				relevant_docs = labels_json[label]
				rel_cases += len(relevant_docs)
				for i in range(len(relevant_docs)):
					relevant_docs[i] = relevant_docs[i].replace(".txt","")
				retreived = []
				for line in results_lines:
					line_split = line.split('\t')
					if(line_split[0]==label.replace(".txt","")):
						retreived.append(line_split[2])
			
				for i in range(min(topn, len(retreived))):
					if(retreived[i] in relevant_docs):
						rel_num+=1
		
		else:
			results_lines = rf.readlines()
			labels_txt = lf.readlines()

			rel_num = 0
			rel_cases = 0
			for i in range(0, len(labels_txt), 5):
				label = labels_txt[i].split(' ')[0]
				relevant_docs = [labels_txt[i].split(' ')[2], labels_txt[i+1].split(' ')[2], labels_txt[i+2].split(' ')[2] ,labels_txt[i+3].split(' ')[2] ,labels_txt[i+4].split(' ')[2]]
				rel_cases += len(relevant_docs)
				retreived = []
				for line in results_lines:
					line_split = line.split('\t')
					if(line_split[0]==label):
						retreived.append(line_split[2])
			
				for i in range(min(topn, len(retreived))):
					if(retreived[i] in relevant_docs):
						rel_num+=1

		recall = rel_num/rel_cases
		print("recall at top "+str(topn)+" found is", recall)
		return recall

	def calculatePrecision(self, results, labels, topn):
		lf = open(labels, "r")
		rf = open(results, "r")
		if(self.datasetName=="COLIEE2021"):
			results_lines = rf.readlines()
			labels_json = json.load(lf)
			rel_num = 0
			num_queries = 0
			for label in labels_json.keys():
				num_queries +=1
				relevant_docs = labels_json[label]
				for i in range(len(relevant_docs)):
					relevant_docs[i] = relevant_docs[i].replace(".txt","")
				retreived = []
				for line in results_lines:
					line_split = line.split('\t')
					if(line_split[0]==label.replace(".txt","")):
						retreived.append(line_split[2])
			
				for i in range(min(topn, len(retreived))):
					if(retreived[i] in relevant_docs):
						rel_num+=1

		else:
			results_lines = rf.readlines()
			labels_txt = lf.readlines()
			rel_num = 0
			num_queries = 0
			for i in range(0, len(labels_txt), 5):
				num_queries +=1
				label = labels_txt[i].split(' ')[0]
				relevant_docs = [labels_txt[i].split(' ')[2], labels_txt[i+1].split(' ')[2], labels_txt[i+2].split(' ')[2] ,labels_txt[i+3].split(' ')[2] ,labels_txt[i+4].split(' ')[2]]
				
				retreived = []
				for line in results_lines:
					line_split = line.split('\t')
					if(line_split[0]==label):
						retreived.append(line_split[2])
			
				for i in range(min(topn, len(retreived))):
					if(retreived[i] in relevant_docs):
						rel_num+=1

		retreived_cases = num_queries*topn
		precision = rel_num/retreived_cases
		print("precision at top "+str(topn)+" found is", precision)
		return precision



	def calculateF1(self, results, labels, topn):
		recall = PriorCaseRank.calculateRecall(self, results, labels, topn)
		precision = PriorCaseRank.calculatePrecision(self, results, labels, topn)
		f1 = (2*precision*recall)/(precision+recall)
		print("f1 at top "+str(topn)+" found is", f1)
		return f1

	@staticmethod
	def isReapeatedSpace(txt):

		spaces = ["\n", "\xa0"]
		combs = []
		for i in spaces:
			if(i in txt):
				return True
		return False


	@staticmethod
	def read_and_parse_stopwords(self):

		# Open file from string 
		file = open(self.stopwords, "r")

		# Read from provided stopwords file
		raw_data_stopwords = file.read()

		# Assign list of stopwords
		self.stopwordLst = raw_data_stopwords.replace('\t', '\n').split('\n')

		file.close()

		pass

	@staticmethod
	def cleanText(self,txt, romanNumerals=False):

		tokens = []
		my_doc = nlp(txt)
		for token in my_doc:
			tokens.append(token.text)

		
		lst = []
		lst2 = []
		for word in tokens:
	
			if(word in self.suppressedCitationWords):
				lst.append(word)
			elif(romanNumerals and (word in self.romanNumerals)):
				lst2.append(word)
			else:
				low = word.lower()
				lem = wn.lemmatize(low)

				if (lem not in self.stopwordLst) and (lem not in self.customStopwordLst) and (lem not in list(string.punctuation)) and (not PriorCaseRank.isReapeatedSpace(lem)):
					lst.append(lem)
					lst2.append(lem)
	  
		
		return lst, lst2

	def cleanTextNounVerbs(self, txt):

		sp = spacy.load("en_core_web_sm")

		sentences = nltk.sent_tokenize(txt)
		lst = []
		lst2 = []
		for sentence in sentences:
			for token in sp(sentence):
				if(token.text in self.suppressedCitationWords):
					lst.append(token.text)
				elif(token.pos_=="NOUN" or token.pos_=="ADJ" or token.pos_=="VERB"):
					lst.append(wn.lemmatize(token.text.lower()))
					lst2.append(wn.lemmatize(token.text.lower()))
		return lst, lst2


	def createSentences(self,txt):


		sentences = nltk.sent_tokenize(txt)
		
		return [sentence for sentence in sentences]
			
		
	@staticmethod
	def createCaseCategoryDataframe(self, caseCategory, encodingType):

		files = []
		filenames = []
		for filename in os.listdir(caseCategory):
			with open(os.path.join(caseCategory,  filename), encoding=encodingType) as f:
				files.append(f.read())
				filenames.append(filename)
		
		if(caseCategory==self.documentFile):
			self.documentDataFrame = pd.DataFrame({'case_number': [i.replace(".txt","") for i in filenames], 'content': files})
			return self.documentDataFrame
		elif(caseCategory==self.queryFile):
			self.queryDataFrame = pd.DataFrame({'case_number': [i.replace(".txt","") for i in filenames], 'content': files})
			return self.queryDataFrame

	def createDataframes(self):
		""" recomended to use windows-1252 for FIRE2017 and utf-8 for COLIEE2021 """


		if(self.datasetName=="FIRE2017"):
			pickleFilePath1 = self.pickleFilePath1_FIRE2017
			pickleFilePath2 = self.pickleFilePath2_FIRE2017
			encoding = "windows-1252"
		elif(self.datasetName=="COLIEE2021"):
			pickleFilePath1 = self.pickleFilePath1_COLIEE2021
			pickleFilePath2 = self.pickleFilePath2_COLIEE2021
			encoding = "utf-8"
		else:
			print("ERROR dataset name not found")
		
		PriorCaseRank.read_and_parse_stopwords(self)
	
		PriorCaseRank.createCaseCategoryDataframe(self, self.documentFile, encoding)
		PriorCaseRank.createCaseCategoryDataframe(self, self.queryFile, encoding)
		

		self.queryDataFrame['content_clean_with_fragments'], self.queryDataFrame['content_clean'] = zip(*self.queryDataFrame['content'].swifter.apply(lambda x: PriorCaseRank.cleanText(self,x)))
		self.queryDataFrame['content_sentences'] = self.queryDataFrame['content'].swifter.apply(lambda x: PriorCaseRank.createSentences(self, x))
		#self.queryDataFrame['content_pos_with_fragments'], self.queryDataFrame['content_pos'] = zip(*self.queryDataFrame['content'].swifter.apply(lambda x: PriorCaseRank.cleanTextNounVerbs(self, x)))
		print("DONE")
		self.documentDataFrame['content_clean_with_fragments'], self.documentDataFrame['content_clean'] = zip(*self.documentDataFrame['content'].swifter.apply(lambda x: PriorCaseRank.cleanText(self,x)))
		self.documentDataFrame['content_sentences'] = self.documentDataFrame['content'].swifter.apply(lambda x: PriorCaseRank.createSentences(self,x))
		#self.documentDataFrame['content_pos_with_fragments'], self.documentDataFrame['content_pos'] = zip(*self.documentDataFrame['content'].swifter.apply(lambda x: PriorCaseRank.cleanTextNounVerbs(self, x)))

		
	
		print(self.queryDataFrame['content_clean'].head())
		#print(self.queryDataFrame['content_pos'].head())
		
		

		self.queryDataFrame.to_pickle(pickleFilePath1)
		self.documentDataFrame.to_pickle(pickleFilePath2)

		"""
		# tests for data cleaning
		test1 = PriorCaseRank.cleanText(self,self.queryDataFrame['content'][3])

		test2 = PriorCaseRank.cleanText(self,self.queryDataFrame['content'][3])
		
		print(test1)
		"""
	
		
	
	def preProcess(self):

		if(self.datasetName=="FIRE2017"):
			pickleFilePath1 = self.pickleFilePath1_FIRE2017
			pickleFilePath2 = self.pickleFilePath2_FIRE2017
		elif(self.datasetName=="COLIEE2021"):
			pickleFilePath1 = self.pickleFilePath1_COLIEE2021
			pickleFilePath2 = self.pickleFilePath2_COLIEE2021
		else:
			print("ERROR dataset name not found")

		self.queryDataFrame = pd.read_pickle(pickleFilePath1)
		self.documentDataFrame = pd.read_pickle(pickleFilePath2)

		self.documentDataFrame.set_index("case_number", inplace=True)




	@staticmethod
	def results(self, docNum, rankedDocs, resultFile, topn=100):
		testQuerieNum = self.queryDataFrame['case_number'][docNum]

		for x in range(min(topn, len(rankedDocs))):
			rank = str(x + 1)
			docID, score = rankedDocs[x]
			resultFile.write(testQuerieNum + "\tQ0\t" + str(docID) +
			        "\t" + rank + "\t" + str(score) + "\tmyRun\n")
			pass
		pass 


	@staticmethod
	def findRemovedCitationsInQuery(self, query, length):

		queryList = []

		for i in range(len(query)):
			
			if(query[i] in self.suppressedCitationWords):
				queryTrunc = []
				counter1 = 1
				counter2 = 0
				while(counter2<(length//2) and (i-counter1>=0)):
					if(query[i-counter1] not in self.suppressedCitationWords):
						queryTrunc.insert(0, query[i-counter1])
						counter2+=1
					counter1+=1
				
				counter1 = 1
				counter2 = 0
				while(counter2<(length//2) and (i+counter1<len(query))):
					if(query[i+counter1] not in self.suppressedCitationWords):
						queryTrunc.append(query[i+counter1])
						counter2+=1
					counter1+=1
				if(queryTrunc not in queryList):
					queryList.append(queryTrunc)
		return queryList



	@staticmethod
	def truncateBeginning(self, query, length):

		if("summary" in query[:256] and query[query.index("summary"):length]!=['summary', 'case', 'unedited', 'contains', 'summary', 'footnote', 'case', 'appear', 'document']):
			queryBeginning = query[query.index("summary"):length]
			if(queryBeginning[:9]!=['summary', 'case', 'unedited', 'contains', 'summary', 'footnote', 'case', 'appear', 'document']):
				return query[query.index("summary")+1:length]
			elif("introduction" in query[:256]):
				return query[query.index("introduction")+1:length]
			else:
				return query[query.index("summary")+10:length]
		elif("introduction" in query[:256]):
			return query[query.index("introduction")+1:length]
		else:
			return query[:length]

	@staticmethod
	def truncateEnd(self, query, length):
		"""
		if(len(query)>2*length):
			return query[len(query)-length:]
		
		elif(len(query)>length):
			return query[length:]
		else:
			return []
		"""
		return query[len(query)-length:]

	@staticmethod
	def truncate(self, query, length):

		return PriorCaseRank.truncateBeginning(self, query, length) + PriorCaseRank.truncateEnd(self, query, length)

	@staticmethod
	def rankDocs(self, testQuerie):
		doc_scores = self.bm25.get_scores(testQuerie)
	
		x=0
		dictSum = {}
		for sim in doc_scores:
			dictSum[self.documentDataFrame.index[x]] = sim			
			x += 1

		rankedDocs = [(k, v) for k, v in sorted(dictSum.items(), key=lambda item: item[1], reverse=True)]
		
		return rankedDocs

	def bm25Rank(self, content):
		self.bm25 = BM25Plus(self.documentDataFrame[content])
		results = open(self.resultFile, 'w+')
		querySize =self.queryDataFrame.count()[content]
		for queryNum in tqdm(range(querySize)):
		#for queryNum in tqdm(range(10)):
			rankedDocs = PriorCaseRank.rankDocs(self, self.queryDataFrame[content][queryNum])
			#rankedDocs=PriorCaseRank.pseudoRelevanceFeedback(self, rankedDocs,self.queryDataFrame['content_clean'][queryNum])
			PriorCaseRank.results(self, queryNum, rankedDocs, results)
		
		if(self.datasetName=="FIRE2017"):
			subprocess.run("./trec_eval -q -m num_q -m map -m P.10 -m num_rel -m num_rel_ret ../FIRE2017-IRLeD-track-data/Task_2/irled-qrel.txt ../Results.txt > ../evaluation_FIRE2017_bm25Plus_"+content+".txt", shell=True, cwd='./trec_eval-9.0.7')
		elif(self.datasetName=="COLIEE2021"):
			if(self.test):
				subprocess.run("./trec_eval -q -m num_q -m map -m P.10 -m num_rel -m num_rel_ret ../"+self.qrel+" ../Results.txt > ../evaluation_COLLIE2021_bm25Plus_"+content+".txt", shell=True, cwd='./trec_eval-9.0.7')
		else:
			print("ERROR dataset name not found")
		
		results.close()
		

	@staticmethod
	def rankDocsWithDivision(self, testQuerie, fullQuery, sumAnalysis):


		docScoreList = []
		for query in testQuerie:
			scores = self.bm25.get_scores(query)
			docScoreList.append(scores)
		
		if(len(docScoreList)>0):
			if(sumAnalysis):
				doc_scores = docScoreList[0]
				for i in range(1, len(docScoreList)):
					doc_scores = numpy.add(doc_scores, docScoreList[i])
			else:
				doc_scores = docScoreList[0]
				for i in range(len(doc_scores)):
					scores = []
					for doc in docScoreList:
						scores.append(doc[i])
					doc_scores[i] = max(scores)


			''' returns ranked by score dictionary with doc id as key and score as value '''
			x=0
			dictSum = {}
			for sim in doc_scores:

				dictSum[self.documentDataFrame.index[x]] = sim
				
				x += 1

			rankedDocs = [(k, v) for k, v in sorted(dictSum.items(), key=lambda item: item[1], reverse=True)]
			
			return rankedDocs
		else:
			return PriorCaseRank.rankDocs(self, fullQuery)




	def bm25DividedRank(self, content, lengthOfFragments, sumAnalysis):
		print(self.documentDataFrame[content].shape)
		self.bm25 = BM25Plus(self.documentDataFrame[content])
		results = open(self.resultFile, 'w+')
		querySize =self.queryDataFrame.count()[content]
		for queryNum in tqdm(range(querySize)):
		#for queryNum in tqdm(range(10)):
			query = self.queryDataFrame[content+"_with_fragments"][queryNum]
			queryFrag = PriorCaseRank.findRemovedCitationsInQuery(self,query, lengthOfFragments)
			rankedDocs = PriorCaseRank.rankDocsWithDivision(self, queryFrag, self.queryDataFrame[content][queryNum], sumAnalysis)
			PriorCaseRank.results(self, queryNum, rankedDocs, results)
		
		if(self.datasetName=="FIRE2017"):
			subprocess.run("./trec_eval -q -m num_q -m map -m P.10 -m num_rel -m num_rel_ret ../FIRE2017-IRLeD-track-data/Task_2/irled-qrel.txt ../Results.txt > ../evaluation_FIRE2017_bm25Plus_"+content+"_divided_"+str(lengthOfFragments)+ "_" +str(sumAnalysis)+".txt", shell=True, cwd='./trec_eval-9.0.7')[1:]
		elif(self.datasetName=="COLIEE2021"):
			subprocess.run("./trec_eval -q -m num_q -m map -m P.10 -m num_rel -m num_rel_ret ../"+self.qrel+" ../Results.txt > ../evaluation_COLLIE2021_bm25Plus_"+content+"_divided_"+str(lengthOfFragments)+ "_" +str(sumAnalysis)+".txt", shell=True, cwd='./trec_eval-9.0.7')
		else:
			print("ERROR dataset name not found")
		
		results.close()


	def bm25SummaryRank(self, content, lengthOfFragments):
		
		print("truncating documents...")
		self.documentDataFrame[content+"_truncated"] = self.documentDataFrame[content].swifter.apply(lambda x: PriorCaseRank.truncateBeginning(self,x, lengthOfFragments))
		print("done!")
		
		start = time.time()
			
		self.bm25 = BM25Plus(self.documentDataFrame[content+"_truncated"])
		results = open(self.resultFile, 'w+')
		querySize =self.queryDataFrame.count()[content]
		for queryNum in tqdm(range(querySize)):
		#for queryNum in tqdm(range(10)):
			query = self.queryDataFrame[content][queryNum]
			queryTrunc = PriorCaseRank.truncateBeginning(self,query, lengthOfFragments) 
			#queryTrunc2 = PriorCaseRank.truncateEnd(self,query, lengthOfFragments) 
			#rankedDocs = PriorCaseRank.rankDocsWithDivision(self, [queryTrunc, queryTrunc2], query, sumAnalysis)
			rankedDocs = PriorCaseRank.rankDocs(self, queryTrunc)
			PriorCaseRank.results(self, queryNum, rankedDocs, results)
		
		if(self.datasetName=="FIRE2017"):
			subprocess.run("./trec_eval -q -m num_q -m map -m P.10 ../FIRE2017-IRLeD-track-data/Task_2/irled-qrel.txt ../Results.txt > ../evaluation_FIRE2017_bm25Plus_"+content+"_summary_"+str(lengthOfFragments)+ "_" +str(sumAnalysis)+ ".txt", shell=True, cwd='./trec_eval-9.0.7')
		elif(self.datasetName=="COLIEE2021"):
			subprocess.run("./trec_eval -q -m num_q -m map -m P.10 -m num_rel -m num_rel_ret ../"+self.qrel+" ../Results.txt > ../evaluation_COLLIE2021_bm25Plus_"+content+"_summary_"+str(lengthOfFragments)+".txt", shell=True, cwd='./trec_eval-9.0.7')
		else:
			print("ERROR dataset name not found")
		
		results.close()

		print("time:", time.time() - start)



	def word2vec_initialLoad(self):

		data = self.documentDataFrame['content_clean'].to_numpy().tolist()

		data = data + self.queryDataFrame['content_clean'].to_numpy().tolist()
		
		cores = multiprocessing.cpu_count()

		self.w2v_model = Word2Vec(data, vector_size=300, min_count=1, workers=cores-1)

		self.w2v_model.save("legalW2V.model")

	def doc2vec_initialLoad(self):

		#self.queryDataFrame.set_index("case_number", inplace=True)

		data = [doc for doc in self.documentDataFrame['content_clean']]
		taggedData = [TaggedDocument(words=self.documentDataFrame['content_clean'][self.documentDataFrame.index[i]], tags=str(i)) for i, val in enumerate(data)]
		print(taggedData[0])
		cores = multiprocessing.cpu_count()

		self.d2v_model = Doc2Vec(vector_size=50, min_count=2, workers=cores-1, epochs=40)

		self.d2v_model.build_vocab(taggedData)

		self.d2v_model.train(taggedData, total_examples=self.d2v_model.corpus_count, epochs=self.d2v_model.epochs)

		self.d2v_model.save("legalD2V.model")

	def doc2vec_initialLoad_v2(self):

		#self.queryDataFrame.set_index("case_number", inplace=True)

		data = [doc for doc in self.documentDataFrame['content_clean']] + [doc for doc in self.queryDataFrame['content_clean']]
		taggedData = [TaggedDocument(words=val, tags=str(i)) for i, val in enumerate(data)]
		print(taggedData[0])
		cores = multiprocessing.cpu_count()

		self.d2v_model = Doc2Vec(vector_size=50, min_count=2, workers=cores-1, epochs=40)

		self.d2v_model.build_vocab(taggedData)

		self.d2v_model.train(taggedData, total_examples=self.d2v_model.corpus_count, epochs=self.d2v_model.epochs)

		self.d2v_model.save("legalD2V_v2.model")


	def word2vec_preprocess(self):

		self.w2v_model = Word2Vec.load("legalW2V.model")

	def doc2vec_preprocess(self):

		self.d2v_model = Doc2Vec.load("legalD2V.model")


	def doc2vec_preprocess_v2(self):

		self.d2v_model = Doc2Vec.load("legalD2V_v2.model")



	@staticmethod
	def removeDuplicatesForCosineSimilarity(self, similarity, docIds):
		scores = []
		for i in range(len(similarity)):
			scores.append((docIds[i], similarity[i]))
		docIdUsed = []
		scoresWithoutDuplicates = {}
		for i in scores:
			if(i[0] in docIdUsed):
				if(i[1]>scoresWithoutDuplicates[i[0]]):
					scoresWithoutDuplicates[i[0]]=i[1]
			else:
				scoresWithoutDuplicates[i[0]] = i[1]
				docIdUsed.append(i[0])
		return scoresWithoutDuplicates


	@staticmethod
	def findRemovedCitationsInQuerySntences(self, query, length):

		queryList = []

		for i in range(len(query)):
			boo = False
			for j in self.suppressedCitationWords:
				if(j in query[i]):
					boo = True
					break
			if(boo):
				queryTrunc = []
				counter1 = 1
				counter2 = 0
				while(counter2<(length//2) and (i-counter1>=0)):
					sentClean = query[i-counter1]
					for cit in self.suppressedCitationWords:
						sentClean = sentClean.replace(cit, "")
					queryTrunc.insert(0, sentClean)
					counter2+=1
					counter1+=1
				
				counter1 = 1
				counter2 = 0
				while(counter2<(length//2) and (i+counter1<len(query))):
					sentClean = query[i+counter1]
					for cit in self.suppressedCitationWords:
						sentClean = sentClean.replace(cit, "")
					
					queryTrunc.append(sentClean)
					counter2+=1
					counter1+=1
				if(queryTrunc not in queryList):
					queryList.append(queryTrunc)
		return queryList


	@staticmethod
	def cleanSentences(self, sents):

		lst = []
		lst2 = []
		self.read_and_parse_stopwords(self)
		customStopwordLst2 = ["\n", "'s", "...", "\n ", " \n", " \n ", "\xa0", "date_suppresed"]
		for sent in sents:
			sentClean1 = []
			sentClean2 = []
			newSent = sent

			for stop in customStopwordLst2:
				newSent = newSent.replace(stop, " ")

			sentSplit = newSent.split(" ")
			for i in range(len(sentSplit)):
				boo = False
				for cit in self.suppressedCitationWords:
					if(cit in sentSplit[i]):	
						sentClean1.append(sentSplit[i])
						boo = True
						break
				#if(sentSplit[i].lower() not in self.stopwordLst) and (boo==False) and (sentSplit[i].lower() not in list(string.punctuation)):
				if(boo==False) and (sentSplit[i].lower() not in list(string.punctuation)):
					sentClean1.append(sentSplit[i].lower())
					sentClean2.append(sentSplit[i].lower())

			lst.append(" ".join(sentClean1))
			lst2.append(" ".join(sentClean2))

		return lst, lst2

	def BertEmbedingRank_bm25(self, content, content2, lengthOfFragments, topn, sumAnalysis):

		model = SentenceTransformer('nlpaueb/legal-bert-base-uncased')
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		torch.cuda.empty_cache()
		model.to(device)


		print("cleaning sentences...")
		self.documentDataFrame[content2+"_with_fragments"], self.documentDataFrame[content2+"_clean"] = zip(*self.documentDataFrame[content2].swifter.apply(lambda x: self.cleanSentences(self,x)))
		self.queryDataFrame[content2+"_with_fragments"] , self.queryDataFrame[content2+"_clean"] = zip(*self.queryDataFrame[content2].swifter.apply(lambda x: self.cleanSentences(self,x)))
		print("done!")
		"""
		print("truncating documents...")
		self.documentDataFrame[content+"_truncated"] = self.documentDataFrame[content].swifter.apply(lambda x: PriorCaseRank.truncateBeginning(self,x, 512))
		print("done!")
		"""
		self.bm25 = BM25Plus(self.documentDataFrame[content])
		results = open(self.resultFile, 'w+')
		querySize =self.queryDataFrame.count()[content]

		
		for queryNum in tqdm(range(querySize)):
			query = self.queryDataFrame[content][queryNum]
			#queryTrunc = PriorCaseRank.truncateBeginning(self,query, 512) 
			#rankedDocs = PriorCaseRank.rankDocs(self, queryTrunc)
			queryDivided= PriorCaseRank.findRemovedCitationsInQuery(self, self.queryDataFrame[content+"_with_fragments"][queryNum], lengthOfFragments)
			rankedDocs = PriorCaseRank.rankDocsWithDivision(self, queryDivided, query, sumAnalysis)

			queryDivided = self.findRemovedCitationsInQuerySntences(self, self.queryDataFrame[content2+"_with_fragments"][queryNum], 30)
			if(len(queryDivided)>0):
				for i in range(len(queryDivided)):
					queryDivided[i] = ' '.join(queryDivided[i])
				queryLength = len(queryDivided)
				docStrings = queryDivided
				docIds = []
				for item in rankedDocs[:topn]:
					doc = self.documentDataFrame[content2+"_clean"][item[0]]
					for i in range(0, len(list(' '.join(doc))), 512):
						frag = ' '.join(doc[i:i+512])	
						docStrings.append(frag)
						docIds.append(item[0])
				
				doc_embeddings = model.encode(docStrings)
				similarity = cosine_similarity(doc_embeddings[:queryLength], doc_embeddings[queryLength:])
			
				scores = [0 for i in range(len(similarity[0]))]
				for i in range(len(similarity)):
					for j in range(len(similarity[i])):
						#scores[j] = scores[j] + similarity[i][j]
						scores[j] = max(scores[j], similarity[i][j])

				scoresWithoutDuplicates = self.removeDuplicatesForCosineSimilarity(self, scores, docIds)
				rankedDocs = [(k, v) for k, v in sorted(scoresWithoutDuplicates.items(), key=lambda item: item[1], reverse=True)]
			
			self.results(self, queryNum, rankedDocs, results)

		if(self.datasetName=="FIRE2017"):
			subprocess.run("./trec_eval -q -m num_q -m map -m P.10 -m num_rel -m num_rel_ret ../FIRE2017-IRLeD-track-data/Task_2/irled-qrel.txt ../Results.txt > ../evaluation_FIRE2017_BERT_DIVIDED.txt", shell=True, cwd='./trec_eval-9.0.7')
		elif(self.datasetName=="COLIEE2021"):
			subprocess.run("./trec_eval -q -m num_q -m map -m P.10 -m num_rel -m num_rel_ret ../"+self.qrel+" ../Results.txt > ../evaluation_COLLIE2021_BERT_DIVIDED.txt", shell=True, cwd='./trec_eval-9.0.7')
		else:
			print("ERROR dataset name not found")
		
		results.close()





pcr = PriorCaseRank("COLIEE2021")
#pcr.createDataframes()
pcr.preProcess()

pcr.bm25DividedRank('content_clean', 128, True)
pcr.calculateF1("Results_bm25_divided_fullpool.txt", "COLIEE2021/test_labels.json",7)






 ### past function calls
"""


pcr2 = PriorCaseRank("FIRE2017")
pcr2.preProcess()
print()
pcr2.calculateF1("Results_FIRE2017_bm25_divided_true.txt", "FIRE2017-IRLeD-track-data/Task_2/irled-qrel.txt",10)
pcr2.calculateF1("Results_FIRE2017_bm25_divided_false.txt", "FIRE2017-IRLeD-track-data/Task_2/irled-qrel.txt",10)
pcr2.calculateF1("Results_FIRE2017_bm25.txt", "FIRE2017-IRLeD-track-data/Task_2/irled-qrel.txt",10)
"""

#pcr.bm25SummaryRank("content_clean", 512)
#pcr.bm25DividedRank('content_clean', 128, True)
#pcr.bm25DividedRank('content_clean', 128, False)



