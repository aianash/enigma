#recompile and remove old files
make
#train noun phrase model
time ./word2vectrainer -train ../datafiles/preprocesseddata/nounphrases -output ../datafiles/word2vecdata/nounphrasevectors.bin -savevocab ../datafiles/word2vecdata/nounvocab -cbow 1 -vectorsize 200 -window 10 -negativesampling 25 -hiersoftmax 0 -randomsample 1e-4 -iterations 20 -wordfrequency 5 -threads 4
#sleep for 10 secs as thread exit can take a few secs
sleep 5s
#train adjectives model
time ./word2vectrainer -train ../datafiles/preprocesseddata/adjectivephrases -output ../datafiles/word2vecdata/adjectivevectors.bin -savevocab ../datafiles/word2vecdata/adjvocab -cbow 1 -vectorsize 200 -negativesampling 25 -hiersoftmax 0 -randomsample 1e-4 -iterations 100 -threads 1 -contextknown 1 -contextsize 10 -wordfrequency 1