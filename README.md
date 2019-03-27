# data_prepocessing_for_TPM
Data prepocessing for TPM

**Note** This code for Chinese datasets.

This code include two parts:
* lda model training
* write txt file to bin file

## LDA model training
We obtained the topic information from the classical topic model —— Latent Dirichlet Allocation (LDA).

**Note** We use the [`gensim.ldamodel`](https://radimrehurek.com/gensim/models/ldamodel.html) to training the model. Before training, you should do some pre-processing work, like removing Chinese stop words. The file which contains 1893 Chinese stop words is `Chinese_stopwords_1893.txt`.

1. Find the best topic number for LDA model.
In order to improve the performance of the LDA model, we must first specify the number of topics mined from model. you can run `train_sample_ldamodel()` to training a series of LDA models corresponding to different topics in sample datasets. The reason for this is to increase the training speed. Then run `find_best_topic_num(eval_method)` to find the best topic number. Note that you can use the perplexity indicator or the method provided by [`gensim.ldamodel.CoherenceModel`](https://radimrehurek.com/gensim/models/coherencemodel.html)

2. training LDA model in the final training dataset
You can run `train_final_ldamodel(topic_number)` to get LDA model in the final training dataset accordding to the number of best topics based on the first step.

## Write txt to bin
**Note**
* The article and abstarct are placed in different folders, and they are associated by the same file name.
* We divide the dataset into three parts: all_train.txt, all_val.txt and all_test.txt. Each of the txt file contains a series of file name.

This script will do several things:

* The directories `content_stories_tokenized` and `target_stories_tokenized` are filled with tokenized versions of article and abstarct.
* For each of the url lists `all_train.txt`, `all_val.txt` and `all_test.txt`, the corresponding tokenized stories are read from file, lowercased and written to serialized binary files `train.bin`, `val.bin` and `test.bin`. These will be placed in the newly-created `finished_files` directory. This may take some time.
* Additionally, a `vocab` file is created from the training data. This is also placed in `finished_files`.
* Lastly, `train.bin`, `val.bin` and `test.bin` will be split into chunks of 1000 examples per chunk. These chunked files will be saved in `finished_files/chunked` as e.g. `train_000.bin`, `train_001.bin`, ..., `train_287.bin`. This should take a few seconds. You can use either the single files or the chunked files as input to the Tensorflow code (see considerations [here](https://github.com/abisee/cnn-dailymail/issues/3)).