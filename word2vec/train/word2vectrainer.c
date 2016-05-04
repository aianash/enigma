#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define MAX_STRING_LEN 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_CODE_LEN 40
#define MAX_SENTENCE_LENGTH 1000

const int wordVocabHashSize = 10000000;
const int unigramTableSize = 1e8;

struct wordrep {
	long long wordCount;
	int* nodes;
	char* word;
	char* binCode;
	char codelen;
};

char trainingFile[MAX_STRING_LEN];
char outvectorFile[MAX_STRING_LEN];
char saveVocabFile[MAX_STRING_LEN];
char readVocabFile[MAX_STRING_LEN];

int usecbow = 0;
int minRedLimit = 1;
int minWordCount = 1; // remove big noun phrases 
int hierSoftmax = 0;
int negSample = 5;
int numThreads = 5;
int wordWindow = 5;
int storeVecAsBinary = 0;


float alpha = 0.025;
float seedAlpha = 0.0;
float subSampling = 1e-4;

long long vocabMaxSize = 1000;
long long wordVocabSize = 0;
long long numTrainingWords = 0;
long long trainFileSize = 0;
long long layerSize = 100;
long long wordCountReal = 0;
long long iterations = 5;

struct wordrep* wordVocab;

int* wordVocabHash;
int* unigramTable;

float* expTable;
float* alignMem1;
float* alignMem2;
float* alignMem1Neg;

clock_t startTime;	


/* returns index of argument vlue */
int getArgPos(char* arg, int argCount, char** argValues) {
	/*vars*/
	int index;
	/*execution*/
	for (index = 1; index < argCount; index++) {
		if (!strcmp(arg, argValues[index])) {
			if (index == argCount - 1) {
				printf("Argument value missing for %s\n", arg);
				exit(1);
			}
			return index;
		}
	}
	//arg not found
    return -1;
}

/* returns hash for given word*/
int getWordHash(char* word) {
	/*vars*/
	unsigned long long index;
	unsigned long long wordHash = 0;

	/*execution*/
	for(index = 0; index < strlen(word); index++) {
		wordHash = wordHash * 257 + word[index];
	}
	wordHash = wordHash % wordVocabHashSize;
	return wordHash;
}

int addWordInVocab(char* word) {
	/*vars*/
	unsigned int wordHash;
	unsigned int wordLen = strlen(word) + 1;

	/*execution*/
	// truncate word if too long
	if (wordLen > MAX_STRING_LEN) {
		wordLen = MAX_STRING_LEN;
	}
	//add word to vocab
	wordVocab[wordVocabSize].word = (char*) calloc(wordLen, sizeof(char));
	strcpy(wordVocab[wordVocabSize].word, word);
	wordVocab[wordVocabSize].wordCount = 1;
	wordVocabSize++;
	//allocate more space for vocab if necessary
	if (wordVocabSize + 2 >= vocabMaxSize) {
		vocabMaxSize += 1000;
		wordVocab = (struct wordrep*)realloc(wordVocab, 
			vocabMaxSize * sizeof(struct wordrep));
		if (wordVocab == NULL) {
			printf("ERROR: could not allocate memory for word hash!\n");
			// call deinit
			exit(1);
		}
	}
	//use chaining to break hash collision
	wordHash = getWordHash(word);
	while (wordVocabHash[wordHash] != -1) {
		wordHash = (wordHash + 1) % wordVocabHashSize;
	}
    // add vocab location to hash
	wordVocabHash[wordHash] = wordVocabSize - 1;
	return wordVocabSize - 1;
}

/*reads word from a file and return it in word arg.
  delimiters are space, tab and end of line */
void readWord(char* word, FILE* fileIter) {
	/*vars*/
	int index = 0;
	int readChar;

	/*execution*/
	while(!feof(fileIter)) {
		readChar = fgetc(fileIter);
		//continue on carriage return
		if (readChar == 13) {
			continue;
		}
		if ((readChar == ' ') || (readChar == '\t') || 
			(readChar == '\n')) {
			if (index > 0) {
				if(readChar == '\n') {
					ungetc(readChar, fileIter);
				}
				break;
			}
			if(readChar == '\n') {
				strcpy(word, (char*) "</s>");
				return;
			} else {
				continue;
			}
		}
		word[index++] = readChar;
		if (index >= MAX_STRING_LEN - 1) {
			index--;
		}
	}
	//add end character
	word[index] = 0;
}


/* returns index of word in vocabulary, 
	returns -1 if not found*/
int searchInVocab(char* word) {
	/*vars*/
	unsigned int wordHash = getWordHash(word);

	/*execution*/
	while (1) {
		if (wordVocabHash[wordHash] == -1) {
			//word not found
			return -1;
		}
		if (!strcmp(word, wordVocab[wordVocabHash[wordHash]].word)) {
			return wordVocabHash[wordHash];
		}
		wordHash = (wordHash + 1) % wordVocabHashSize;
	}
	return -1;
}

/*reads a word and returns its index in the vocubalary*/
int getWordIndex(FILE* fileIter) {
	/*vars*/
	char word[MAX_STRING_LEN];

	/*execution*/
	readWord(word, fileIter);
	if(feof(fileIter)) {
		return -1;
	}
	return searchInVocab(word);
}

/*reduce size of vocabulary when it becomes large by removing
  infrequent words*/
void reduceWordVocab() {
	/*vars*/
	int index1 = 0;
	int index2 = 0;
	unsigned int wordHash;

	/*execution*/
	for(index1 = 0; index1 < wordVocabSize; index1++) {
		if(wordVocab[index1].wordCount > minRedLimit) {
			wordVocab[index2].wordCount = wordVocab[index1].wordCount;
			wordVocab[index2].word = wordVocab[index1].word;
			index2++;
		} else {
			free(wordVocab[index1].word);
		}
	}
	//new vocabulary size
	wordVocabSize = index2; 
	//calculate new hashes
	for(index1 = 0; index1 < wordVocabHashSize; index1++) {
		wordVocabHash[index1] = -1;
	}
	for(index1 = 0; index1 < wordVocabSize; index1++) {
		wordHash = getWordHash(wordVocab[index1].word);
		while (wordVocabHash[wordHash] != -1) {
			wordHash = (wordHash + 1) % wordVocabHashSize;
		}
		wordVocabHash[wordHash] = index1;
	}
	//increase limit for next iteration
	minRedLimit++;

}

/*comparator for sorting vocab in decreasing order of word frequency*/
int wordVocabComparator(const void* a, const void* b) {
	return ((struct wordrep*)b)->wordCount - ((struct wordrep*)a)->wordCount;
}

/*sort the vocabulary by frequency using word counts*/
void sortWordVocab() {
	/*vars*/
	int index = 0;
	int size = 0;
	unsigned int wordHash;

	/*execution*/
	//check for position 0 after reduction
	//keep first one as </s>
	qsort(&wordVocab[1], wordVocabSize - 1, sizeof(struct wordrep),
			wordVocabComparator);
	//recalculate hash
	for(index = 0; index < wordVocabHashSize; index++) {
		wordVocabHash[index] = -1;
	}
	size = wordVocabSize;
	numTrainingWords = 0;
	for(index = 0; index < size; index++) {
		//word whose frequency is less than minWordCount is discarded
		if ((wordVocab[index].wordCount < minWordCount) && (index != 0)) {
			wordVocabSize--;
			free(wordVocab[index].word);
		} else {
			wordHash = getWordHash(wordVocab[index].word);
			while (wordVocabHash[wordHash] != -1) {
				wordHash = (wordHash + 1) % wordVocabHashSize;
			}
			wordVocabHash[wordHash] = index;
			numTrainingWords += wordVocab[index].wordCount;
		}
	}
	//reduce memory allocated after infrequent words removed
	wordVocab = (struct wordrep*)realloc(wordVocab, 
					(wordVocabSize + 1) * sizeof(struct wordrep));
}

/*learn vocabulary from training file*/
void learnWordVocab() {
	/*vars*/
	char word[MAX_STRING_LEN];
	FILE* trainFileIter;
	long long index;

	/*execution*/
	//initialize hash
	for(index = 0; index < wordVocabHashSize; index++) {
		wordVocabHash[index] = -1;
	}
	//open train file
	trainFileIter = fopen(trainingFile, "rb");
	if (trainFileIter == NULL) {
		printf("ERROR: training file missing! \n");
		//call deinit
		exit(1);
	}
	wordVocabSize = 0;
	//add start indicator to vocab
	addWordInVocab((char*) "</s>");
	//read training file and fill hash
	while(1) {
		readWord(word, trainFileIter);
		//printf("%s\n", word);
		if (feof(trainFileIter)) {
			//check if last word is captured
			break;
		}
		//got a new word
		numTrainingWords++;
		//search in hash
		index = searchInVocab(word);
		if (index == -1) {
			//new word
		    addWordInVocab(word);
		} else {
			//already found that word earlier
			wordVocab[index].wordCount++;
		}
        //implement reduce vocab by removing infrequent words
        if(wordVocabSize > wordVocabHashSize * 0.8) {
         	reduceWordVocab();
        }
	}
	//get training file size
	trainFileSize = ftell(trainFileIter);
	//sort word vocabulary based on word count frequency
	sortWordVocab();
	printf("Vocab size: %lld\n", wordVocabSize);
	printf("Words in train file: %lld\n", numTrainingWords);
	fclose(trainFileIter);
}

/*read vocbulary from saved vocab file*/
void readWordVocab() {
	/*vars*/
	long long index = 0;
	char readChar;
	char word[MAX_STRING_LEN];
	FILE* trainFileIter;
	FILE* vocabFileIter; 

	/*execution*/
	vocabFileIter = fopen(readVocabFile, "rb");
	if (vocabFileIter == NULL) {
		printf("ERROR: Vocabulary file missing!\n");
		exit(1);
	}
	for(index = 0; index < wordVocabHashSize; index++) {
		wordVocabHash[index] = -1;
	}
	wordVocabSize = 0;
	//read vocabulary from file
	while(1) {
		readWord(word, vocabFileIter);
		if (feof(vocabFileIter)) {
			break;
		}
		index = addWordInVocab(word);
		//read newline to reach next line in next iteration
		fscanf(vocabFileIter, "%lld%c", &wordVocab[index].wordCount, &readChar);
	}
	// sort vocab according to word frequency
	sortWordVocab();
	printf("Vocab size: %lld\n", wordVocabSize);
	printf("Words in training file: %lld\n", numTrainingWords);
	fclose(vocabFileIter);
	//get size of training file
	trainFileIter = fopen(trainingFile, "rb");
	if (trainFileIter == NULL) {
		printf("ERROR: traing file missing!\n");
		exit(1);
	}
	fseek(trainFileIter, 0, SEEK_END);
	trainFileSize = ftell(trainFileIter);
	fclose(trainFileIter);
}

/*save vocabulary in file*/
void saveWordVocab() {
	/*vars*/
	long long index;
	FILE* vocabFileIter;

	/*execution*/
	vocabFileIter = fopen(saveVocabFile, "wb");
	if (vocabFileIter == NULL) {
		printf("ERROR: vocabulary file missing!\n");
		exit(1);
	}
	for(index = 0; index < wordVocabSize; index++) {
		// printf("%s %lld\n", wordVocab[index].word,
		// 	wordVocab[index].wordCount);
		fprintf(vocabFileIter, "%s %lld\n", wordVocab[index].word,
			wordVocab[index].wordCount);	
	}
	fclose(vocabFileIter);
}

//create binary huffman code tree accordind to word frequeny in 
//vocabulary. Frequentwords will have short binary codes
void createBinaryTree() {
	/*vars*/
	long long* count = (long long*)calloc(wordVocabSize * 2 + 1, sizeof(long long));
	long long* binary = (long long*)calloc(wordVocabSize * 2 + 1, sizeof(long long));
	long long* parentNode = (long long*)calloc(wordVocabSize * 2 + 1, sizeof(long long));

	long long index1;
	long long index2;
	long long index3;
	long long pos1;
	long long pos2;
	long long min1;
	long long min2;

	long long nodes[MAX_CODE_LEN];

	char binCode[MAX_CODE_LEN];

    /*execution*/
	//allocate memory for binary tree construction
	for(index1 = 0; index1 < wordVocabSize; index1++) {
		wordVocab[index1].binCode = (char*)calloc(MAX_CODE_LEN, sizeof(char));
		wordVocab[index1].nodes = (int*)calloc(MAX_CODE_LEN, sizeof(int));
	}
	for(index1 = 0; index1 < wordVocabSize; index1++) {
		count[index1] = wordVocab[index1].wordCount;
	}
	for(index1 = wordVocabSize; index1 < wordVocabSize * 2; index1++) {
		count[index1] = 1e15;
	}
	// initial positions 
	pos1 = wordVocabSize - 1;
	pos2 = wordVocabSize;
	//add one node at a time
	for(index1 = 0; index1 < wordVocabSize - 1; index1++) {
		//find two smallest nodes 
		if (pos1 >= 0) {
			if(count[pos1] < count[pos2]) {
				min1 = pos1;
				pos1--;
			} else {
				min1 = pos2;
				pos2++;
			}
		} else {
			min1 = pos2;
			pos2++;
		}
		//find second min node
		if (pos1 >= 0) {
			if(count[pos1] < count[pos2]) {
				min2 = pos1;
				pos1--;
			} else {
				min2 = pos2;
				pos2++;
			}
		} else {
			min2 = pos2;
			pos2++;
		}
		//upfate parent pointers
		count[wordVocabSize + index1] = count[min1] + count[min2];
		parentNode[min1] = wordVocabSize + index1;
		parentNode[min2] = wordVocabSize + index1;
		binary[min2] = 1;
	}
	// assign binary code to each vocabulary word
	for(index1 = 0; index1 < wordVocabSize; index1++) {
		index2 = index1;
		index3 = 0;
		while(1) {
			binCode[index3] = binary[index2];
			nodes[index3] = index2;
			index3++;

			index2 = parentNode[index2];
			if(index2 == wordVocabSize * 2 - 2) {
				//reached the root 
				break;
			}
		}
		wordVocab[index1].codelen = index3;
		wordVocab[index1].nodes[0] = wordVocabSize - 2;
		//assign code
		for(index2 = 0; index2 < index3; index2++) {
			wordVocab[index1].binCode[index3 - index2 - 1] = binCode[index2];
			wordVocab[index1].nodes[index3 - index2] = nodes[index2] - wordVocabSize;
		}
	} 
	//Huffman tree creation completed
	free(count);
	free(binary);
	free(parentNode);
}

/*Initialize networkmemories*/
void initNetwork() {
	/*vars*/
	long long index1;
	long long index2;

	unsigned long long nextRandNum = 1;

	/*execution*/
	posix_memalign((void**)&alignMem1, 128, 
		(long long)wordVocabSize * layerSize * sizeof(float));
	if (alignMem1 == NULL) {
		printf("ERROR: Aligned memory location 1 failed\n");
		exit(1);
	}
	if(hierSoftmax) {
		posix_memalign((void**)&alignMem2, 128, 
			(long long)wordVocabSize * layerSize * sizeof(float));
		if (alignMem2 == NULL) {
			printf("ERROR: Aligned memory location 2 failed\n");
			exit(1);
		}
		//initialize aligned memory 2
		for(index1 = 0; index1 < wordVocabSize; index1++) {
			for(index2 = 0; index2 < layerSize; index2++) {
				alignMem2[index1 * layerSize + index2] = 0;
			}
		}
	}
	//negative sampling
	if(negSample > 0) {
		posix_memalign((void**)&alignMem1Neg, 128, 
			(long long)wordVocabSize * layerSize * sizeof(float));
		if (alignMem1Neg == NULL) {
			printf("ERROR: Aligned memory allocation for neg failed\n");
			exit(1);
		}
		//initialize aligned memory for negs
		for(index1 = 0; index1 < wordVocabSize; index1++) {
			for(index2 = 0; index2 < layerSize; index2++) {
				alignMem1Neg[index1 * layerSize + index2] = 0;
			}
		}

	}
	//randomly initialize aligned memory 1 vectors
	//using 25214903917 and 11 prime numbers for randomization
	for (index1 = 0; index1 < wordVocabSize; index1++) {
		for (index2 = 0; index2 < layerSize; index2++) {
    		nextRandNum = nextRandNum * (unsigned long long)25214903917 + 11;
    		alignMem1[index1 * layerSize + index2] = (((nextRandNum & 0xFFFF) / (float)65536) - 0.5) / layerSize;
    	}
    }
    // create huffman tree
    createBinaryTree();
}

/*Initialize unigram table*/
void initUnigramTable() {
	/*vars*/
	long long index1;
	long long index2;

	double trainWordsPow = 0;
	double dValue;
	double power = 0.75;

	/*execution*/
	unigramTable = (int*)malloc(unigramTableSize * sizeof(int));
	for(index1 = 0; index1 < wordVocabSize; index1++) {
		trainWordsPow += pow(wordVocab[index1].wordCount, power);
	}
	index2 = 0;
	//normalized value
	dValue = pow(wordVocab[index2].wordCount, power) / trainWordsPow;
	//initialize unigram table
	for(index1 = 0; index1 < unigramTableSize; index1++) {
		unigramTable[index1] = index2;
		if(index1 / (double)unigramTableSize > dValue) {
			index2++;
			dValue += pow(wordVocab[index2].wordCount, power) / trainWordsPow;
		}
		if (index2 >= wordVocabSize) {
			index2 = wordVocabSize - 1;
		}
	}
}

/* Function for training in thread*/
void* runTrainThread(void* threadId) {
	/*vars*/
	FILE* trainFileIter;

	long long wordCount = 0;
	long long lastWordCount = 0;
	long long sentenceLength = 0;
	long long wordIndex;
	long long sentence[MAX_SENTENCE_LENGTH + 1];
	long long indexInSentence = 0;
	long long threadLocalIter = iterations;
	long long index1, index2, index3, index4;
	long long contWords;
	long long lastWordIndex;
	long long label;
	long long targetIndex;
	long long updateRow;
	long long updateRow2;

	unsigned long long nextRandNum = (long long)threadId;

	float updateValue;
	float updateGradient;

	clock_t currentTime;

	/*execution*/
	float* net1 = (float*)calloc(layerSize, sizeof(float));
	float* net2 = (float*)calloc(layerSize, sizeof(float));
	//open training file
	trainFileIter = fopen(trainingFile, "rb");
	if (trainFileIter == NULL) {
		printf("ERROR: training file missing!\n");
		exit(1);
	}
	fseek(trainFileIter, trainFileSize / (long long)numThreads * (long long)threadId, 
			SEEK_SET);
	//training loop
	while(1) {
		if (wordCount - lastWordCount > 10000) {
			wordCountReal += wordCount - lastWordCount;
			lastWordCount = wordCount;
			currentTime = clock();
			printf("%c alpha: %f progress: %.2f%% words/thread/sec: %.2f k ",
				13, alpha, wordCountReal / (float)(iterations * numTrainingWords + 1) * 100,
				wordCountReal / ((float) (currentTime - startTime + 1) / (float) CLOCKS_PER_SEC * 1000));
			fflush(stdout);
			alpha = seedAlpha * (1 - wordCountReal / (float)(iterations * numTrainingWords + 1));
			if (alpha < seedAlpha * 0.0001) {
				alpha = seedAlpha * 0.0001;
			}
		}
		//sentence cration loop
		if(sentenceLength == 0) {
			while(1) {
				wordIndex = getWordIndex(trainFileIter);
				if (feof(trainFileIter)) {
					break;
				}
				if(wordIndex == -1) {
					//word not found in hash
					//may be removed as infrequent
					continue;
				}
				wordCount++;
				if(wordIndex == 0) {
					break;
				}
				//subsampling randomly discards frequent words kepping rank same
				// It hels in better representation of less frequent words
				if(subSampling > 0) {
					float randValue = (sqrt(wordVocab[wordIndex].wordCount / (subSampling * numTrainingWords)) + 1) *
											(subSampling * numTrainingWords) / wordVocab[wordIndex].wordCount;
					nextRandNum = nextRandNum * (unsigned long long)25214903917 + 11;
					if (randValue < (nextRandNum & 0xFFFF) / (float)65536) {
						continue;
					}
				}
				sentence[sentenceLength] = wordIndex;
				sentenceLength++;
				if(sentenceLength >= MAX_SENTENCE_LENGTH) {
					break;
				}
			}
			indexInSentence = 0;
		}
		// check for iteration conditions
		if(feof(trainFileIter) || (wordCount > numTrainingWords / numThreads )) {
			wordCountReal += wordCount - lastWordCount;
			threadLocalIter--;
			if (threadLocalIter == 0) {
				//end iterations
				break;
			}
			wordCount = 0;
			lastWordCount = 0;
			sentenceLength = 0;
			fseek(trainFileIter, trainFileSize / (long long)numThreads * (long long)threadId, 
					SEEK_SET);
			continue;
		}
		// start net traing with the sentence received
		wordIndex = sentence[indexInSentence];
		if(wordIndex == -1) {
			//get new sentence 
			//not a vocab word
			continue;
		}
		//initialize networks
		for(index1 = 0; index1 < layerSize; index1++) {
			net1[index1] = 0;
		}
		for(index1 = 0; index1 < layerSize; index1++) {
			net2[index1] = 0;
		}
		nextRandNum = nextRandNum * (unsigned long long)25214903917 + 11;
		index1 = nextRandNum % wordWindow;
		if (usecbow) {
			//train continuous bag of words
			// input -> hidden
			contWords = 0;
			for (index2 = index1; index2 < wordWindow * 2 + 1 - index1; index2++) {
				if (index2 != wordWindow) {
					index3 = indexInSentence - wordWindow + index2;
					if (index3 < 0) {
						continue;
					}
					if(index3 > sentenceLength) {
						continue;
					}
					lastWordIndex = sentence[index3];
					if (lastWordIndex == -1) {
						continue;
					}
					for (index3 = 0; index3 < layerSize; index3++) {
						net1[index3] += alignMem1[index3 + lastWordIndex * layerSize];
					}
					contWords++;
				}	
			}
			if(contWords) {
				for(index3 = 0; index3 < layerSize; index3++) {
					net1[index3] /= contWords;
				}
				// hirarchical softmax
				if(hierSoftmax) {
					for(index4 = 0; index4 < wordVocab[wordIndex].codelen; index4++) {
						updateValue = 0;
						updateRow = wordVocab[wordIndex].nodes[index4] * layerSize;
						//forward hidden -> output
						for (index3 = 0; index3 < layerSize; index3++) {
							updateValue += net1[index3] * alignMem2[index3 + updateRow];
						}
						if(updateValue <= -MAX_EXP) {
							continue;
						} else if (updateValue >= MAX_EXP) {
							continue;
						} else {
							updateValue = expTable[(int)((updateValue + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
						}
						// calculate gradient and multiply by learning rate
						updateGradient = (1 - wordVocab[wordIndex].binCode[index4] - updateValue) * alpha;	
						//backward update output-> hidden
						for(index3 = 0; index3 < layerSize; index3++) {
							net2[index3] += updateGradient * alignMem2[index3 + updateRow];
						}
						//update weights hidden -> output
						for(index3 = 0; index3 < layerSize; index3++) {
							alignMem2[index3 + updateRow] += updateGradient * net1[index3];
						}		
					}	
				} // end of hierchical softmax 
				//negative sampling
				if(negSample > 0) {
					for(index4 = 0; index4 < negSample + 1; index4++) {
						if (index4 == 0) {
							targetIndex = wordIndex;
							label = 1;
						} else {
							nextRandNum = nextRandNum * (unsigned long long)25214903917 + 11;
							targetIndex = unigramTable[(nextRandNum >> 16) % unigramTableSize];
							if (targetIndex == 0) {
								targetIndex = nextRandNum % (wordVocabSize -1) + 1;
							}
							if (targetIndex == wordIndex) {
								continue;
							}
							label = 0;
						}
						updateRow = targetIndex * layerSize;
						updateValue = 0;
						for (index3 = 0; index3 < layerSize; index3++) {
							updateValue += net1[index3] * alignMem1Neg[index3 + updateRow];
						}
						if (updateValue > MAX_EXP) {
							updateGradient = (label - 1) * alpha;
						} else if (updateValue < -MAX_EXP) {
							updateGradient = (label - 0) * alpha;
						} else {
							updateGradient = (label - expTable[(int)((updateValue + MAX_EXP) * 
								(EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
						}
						//propogate errors
						for(index3 = 0; index3 < layerSize; index3++) {
							net2[index3] += updateGradient * alignMem1Neg[index3 + updateRow];
						}
						//learn weights
						for(index3 = 0; index3 < layerSize; index3++) {
							alignMem1Neg[index3 + updateRow] += updateGradient * net1[index3];
						}
					}
				} // end of negative sampling
				// backward hidden -> input layer
				for(index2 = index1; index2 < wordWindow * 2 + 1 - index1; index2++) {
					if (index2 != wordWindow) {
						index3 = indexInSentence - wordWindow + index2;
						if (index3 < 0) {
							continue;
						}
						if (index3 >= sentenceLength) {
							continue;
						}
						lastWordIndex = sentence[index3];
						if(lastWordIndex == -1) {
							continue;
						}
						for(index3 = 0; index3 < layerSize; index3++) {
							alignMem1[index3 + lastWordIndex * layerSize] += net2[index3];
						}
					}
				}
			} // end of continuous word sentence
		} /*end of cbow training */ else {
			// train skipgram
			for(index2 = index1; index2 < wordWindow * 2 + 1 - index1; index2++) {
				if (index2 != wordWindow) {
					index3 = indexInSentence - wordWindow + index2;
					if (index3 < 0) {
						continue;
					}
					if(index3 >= sentenceLength) {
						continue;
					}
					lastWordIndex = sentence[index3];
					if (lastWordIndex == -1) {
						continue;
					}
					updateRow = lastWordIndex * layerSize;
					for(index3 = 0; index3 < layerSize; index3++) {
						net2[index3] = 0;
					}
					//hirarchical softmax
					if(hierSoftmax) {
						for(index4 = 0; index4 < wordVocab[wordIndex].codelen; index4++) {
							updateValue = 0;
							updateRow2 = wordVocab[wordIndex].nodes[index4] * layerSize;
							//propagate hidden->output
							for (index3 = 0; index3 < layerSize; index3++) {
								updateValue += alignMem1[index3 + updateRow] * alignMem2[index3 + updateRow2];
							}
							if (updateValue <= -MAX_EXP) {
								continue;
							} else if (updateValue >= MAX_EXP) {
								continue;
							} else {
								updateValue = expTable[(int)((updateValue + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP/ 2))];
							}
							//update gradient
							updateGradient = (1 - wordVocab[wordIndex].binCode[index4] - updateValue) * alpha;
							//propagate errors
							for (index3 = 0; index3 < layerSize; index3++) {
								net2[index3] += updateGradient * alignMem2[index3 + updateRow2];
							}
							//learn weights
							for (index3 = 0; index3 < layerSize; index3++) {
								alignMem2[index3 + updateRow2] += updateGradient * alignMem1[index3 + updateRow];
							}
						}
					}// end of hier softmax
					//negative sampling
					if(negSample > 0) {
						for (index4 = 0; index4 < negSample + 1; index4++) {
							if (index4 == 0) {
								targetIndex = wordIndex;
								label = 1;
							} else {
								nextRandNum = nextRandNum * (unsigned long long)25214903917 + 11;
								targetIndex = unigramTable[(nextRandNum >> 16) % unigramTableSize];

								if (targetIndex == 0) {
									targetIndex = nextRandNum % (wordVocabSize - 1) + 1;
								}

								if (targetIndex == wordIndex) {
									continue;
								}
								label = 0;
							}
							updateRow2 = targetIndex * layerSize;
							updateValue = 0;
							for(index3 = 0; index3 < layerSize; index3++) {
								updateValue += alignMem1[index3 + updateRow] * alignMem1Neg[index3 + updateRow2];
							}
							if(updateValue > MAX_EXP) {
								updateGradient = (label - 1) * alpha;
							} else if (updateValue < -MAX_EXP) {
								updateGradient = (label - 0) * alpha;
							} else {
								updateGradient = (label - expTable[(int)((updateValue + MAX_EXP) * 
									(EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
							}
							//propagate errors
							for (index3 = 0; index3 < layerSize; index3++) {
								net2[index3] += updateGradient * alignMem1Neg[index3 + updateRow2];
							}
							//learn weights
							for (index3 = 0; index3 < layerSize; index3++) {
								alignMem1Neg[index3 + updateRow2] += updateGradient * alignMem1[index3 + updateRow];
							}	
						}
					} // end of negative sampling
					// learn weights
					for(index3 = 0; index3 < layerSize; index3++) {
						alignMem1[index3 + updateRow] += net2[index3];
					}
				}
			}
		} // end of skipgram training
		//trin next word in sentence
		indexInSentence++;
		if(indexInSentence >= sentenceLength) {
			sentenceLength = 0;
			continue;
		}
	} //end of training loop
	fclose(trainFileIter);
	free(net1);
	free(net2);  //errors
	pthread_exit(NULL);
}

/*Initialize and starts training*/
void trainModel() {
	/*vars*/
	long long index1;
	long long index2;
	pthread_t* threads = (pthread_t*)malloc(numThreads * sizeof(pthread_t));
	seedAlpha = alpha;
	FILE* outvectorFileIter;

	/*execution*/
	printf("Start training word2vec model\n");
	// allocate memory for vocabulary
    wordVocab = (struct wordrep*)calloc(vocabMaxSize, sizeof(struct wordrep));
    wordVocabHash = (int*)calloc(wordVocabHashSize, sizeof(int));
    //precompute exp table for negative sampling
    expTable = (float*)malloc((EXP_TABLE_SIZE + 1) * sizeof(float));
    for (index1 = 0; index1 < EXP_TABLE_SIZE; index1++) {
    	expTable[index1] = exp((index1 / (float)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP);
		expTable[index1] = expTable[index1] / (expTable[index1] + 1);
    }
	//Learchvocab from training file
	if(readVocabFile[0] != 0) {
		//read from vocab file already present
		readWordVocab();
	} else {
		//learn vocabulary from training file
		learnWordVocab();
	}
	if(saveVocabFile[0] != 0) {
		//save vocabulary
		saveWordVocab();
	}
	if (outvectorFile[0] == 0) {
		printf("ERROR: provide output file tto store vectors!\n");
		exit(1);
	}
	//Initialize network  to train word vectors
	initNetwork();
	if(negSample > 0) {
		//negative skipgram sampling power 0.75
		initUnigramTable();
	}
	//get strarting time
	startTime = clock();
	for(index1 = 0; index1 < numThreads; index1++) {
		pthread_create(&threads[index1], NULL, runTrainThread, (void*) index1);
	}
	for(index1 = 0; index1 < numThreads; index1++) {
		pthread_join(threads[index1], NULL);
	}

	outvectorFileIter = fopen(outvectorFile, "wb");
	if (outvectorFileIter == NULL) {
		printf("ERROR: output file missing!\n");
		exit(1);
	}
	//save word vectors
	fprintf(outvectorFileIter, "%lld %lld\n", wordVocabSize, layerSize);
	for(index1 = 0; index1 < wordVocabSize; index1++) {
		fprintf(outvectorFileIter, "%s ", wordVocab[index1].word);
		if(storeVecAsBinary) {
			for (index2 = 0; index2 < layerSize; index2++) {
				fwrite(&alignMem1[index1 * layerSize + index2], 
					sizeof(float), 1, outvectorFileIter);
			}
		} else {
			for(index2 = 0; index2 < layerSize; index2++) {
				fprintf(outvectorFileIter, "%lf ", 
					alignMem1[index1 * layerSize + index2]);
			}
		}

		fprintf(outvectorFileIter, "\n");
	}
	fclose(outvectorFileIter);
	printf("word2vec training complete\n");
}

int main(int argc, char** argv) {
	/*vars*/
	int index = 0;

	/*execution*/
	trainingFile[0] = 0;
	outvectorFile[0] = 0;
	saveVocabFile[0] = 0;
	readVocabFile[0] = 0;
    //read options
	if ((index = getArgPos("-vectorsize", argc, argv)) > 0) {
		layerSize = atoi(argv[ index + 1]);
	}
	if ((index = getArgPos("-train", argc, argv)) > 0) {
		strcpy(trainingFile, argv[index + 1]);
	}
	if ((index = getArgPos("-cbow", argc, argv)) > 0) {
		usecbow = atoi(argv[index + 1]);
		// alpha for cbow = 0.05
		alpha = 0.05;
	}
	if ((index = getArgPos("-alpha", argc, argv)) > 0) {
		alpha = atof(argv[index + 1]);
	}
	if ((index = getArgPos("-output", argc, argv)) > 0) {
		 strcpy(outvectorFile, argv[index + 1]);
	}
	if ((index = getArgPos("-window", argc, argv)) > 0) {
		wordWindow = atoi(argv[index + 1]);
	}
	if ((index = getArgPos("-negativesampling", argc, argv)) > 0) {
		negSample = atoi(argv[index + 1]);
	}
	if ((index = getArgPos("-hiersoftmax", argc, argv)) > 0) {
		hierSoftmax = atoi(argv[index + 1]);
	}
	if ((index = getArgPos("-randomsample", argc, argv)) > 0) {
		subSampling = atof(argv[index + 1]);
	}
	if ((index = getArgPos("-iterations", argc, argv)) > 0) {
		iterations = atoi(argv[index + 1]);
	}
	if ((index = getArgPos("-savevocab", argc, argv)) > 0) {
		strcpy(saveVocabFile, argv[index + 1]);
	}

	printf("\nVector size: %lld\n", layerSize);
	printf("Training file: %s\n", trainingFile);
	printf("Use cbow: %d\n", usecbow);
	printf("Alpha: %0.2f\n", alpha);
	printf("Output file: %s\n", outvectorFile);
	printf("Word window: %d\n", wordWindow);
	printf("Negative sample: %d\n", negSample);
	printf("Heirachical softmax: %d\n", hierSoftmax);
	printf("Random sample: %0.8f\n", subSampling);
	printf("Number of iterations: %lld\n", iterations);
	printf("Saved vocab file: %s\n", saveVocabFile);
	printf("Number of threads: %d\n", numThreads);
	//start trining
    trainModel();
    return 0;
}