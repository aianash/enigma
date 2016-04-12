CC = gcc
#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
CFLAGS = -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result

all: cleandump cleanbin clean word2vec

word2vec : word2vectrainer.c
	$(CC) word2vectrainer.c -o word2vectrainer $(CFLAGS)
	chmod +x *.sh

clean:
	rm -rf word2vectrainer compword2vec

cleandump:
	rm -rf ./dump/*

cleanbin:
	rm -rf ../datafiles/word2vecdata/nounphrasevectors.bin ../datafiles/word2vecdata/nounphrasevectors.t7 