make
time ./word2vectrainer -train ../datafiles/preprocesseddata/wordnetadj -output ../datafiles/word2vecdata/nounphrasevectors.bin -cbow 1 -vectorsize 200 -window 10 -negativesampling 25 -hiersoftmax 0 -randomsample 1e-4 -iterations 100
