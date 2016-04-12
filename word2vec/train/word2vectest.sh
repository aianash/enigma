make
time ./word2vectrainer -train ../datafiles/preprocesseddata/nounphrases -output ../datafiles/word2vecdata/nounphrasevectors.bin -cbow 1 -vectorsize 200 -window 8 -negativesampling 25 -hiersoftmax 0 -randomsample 1e-4 -iterations 15
