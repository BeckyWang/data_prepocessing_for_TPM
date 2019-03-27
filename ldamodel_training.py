# -*- coding: utf-8 -*-
import os
import sys
import math
import numpy as np
import scipy.stats
from time import time
from gensim import corpora
from gensim.models import LdaModel
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary

content_tokenized_dir = 'content_tokenized'

lda_sample_train_ids = './id_lists/lda_sample_train_ids.txt'
lda_sample_train_tokenized_file = 'lda_sample_train_tokenized_file.txt'
lda_sample_result_dir = 'lda_sample_result'

all_test_ids = './id_lists/all_test_ids.txt'
lda_test_tokenized_file = 'lda_test_tokenized_file.txt'

lda_train_ids = './id_lists/all_train_ids.txt'
lda_train_tokenized_file = 'lda_train_tokenized_file.txt'
lda_result_dir = 'lda_result'

def read_text_file(text_file):
  lines = []
  with open(text_file, "r", encoding="utf-8") as f:
    for line in f:
      if line.strip() != '':
        lines.append(line.strip())
  return lines

# 去除停用词，已分好词，空格隔开
def txt_processing(text):
  # 读取停用词表
  stopwords = read_text_file('../Chinese_stopwords_1893.txt')
  # 分词
  wordLst = text.split(' ')
  # 去除停用词
  filtered = [w for w in wordLst if w not in stopwords]
  return ' '.join(filtered)

# 获取文件里id对应的研究内容，预处理后放到一个txt文件中
def get_lda_train_text(id_file_path, result_file_path):
  sample_ids = read_text_file(id_file_path)
  num_samples = len(sample_ids)

  with open(result_file_path, "w", encoding='utf-8') as writer:
    for idx,s in enumerate(sample_ids):
      if idx % 1000 == 0:
          print("正在处理 %i / %i; %.2f已完成" % (idx, num_samples, float(idx)*100.0/float(num_samples)))

      if s.startswith('general_app'):
        if os.path.isfile(os.path.join(content_tokenized_dir, s+'.txt')):
          content_file_path = os.path.join(content_tokenized_dir, s+'.txt')
        else:
          raise Exception('找不到项目%s对应的研究内容！' % s)
        content_file = read_text_file(content_file_path)
        processed_line = txt_processing(' '.join(content_file))

        if idx == num_samples:
          writer.write(processed_line)
        else:
          writer.write(processed_line+'\n')
      else:
        print('不合法的项目id：%s' % s)

# 加载与处理好的文件，返回[[], [], ...]
def load_text_pre(input_file):
  docList = []
  with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
      if line.strip() != '':
        docList.append(line.strip().split())
  return docList

# get training dictionary and corpus with corpora
def get_dic_and_corpus(train_type, result_dir, tokenized_file):
  train = load_text_pre(tokenized_file)
  print("Finished reading processed training data!")

  dictionary = corpora.Dictionary(train)
  # Remove common words that only appear once and appear in more than 50% of documents
  dictionary.filter_extremes(no_below=2, no_above=0.5)
  print('size of dictionary: %i' % len(dictionary.keys()))
  dictionary.save(result_dir + '/' + train_type +'_dictionary.dictionary')

  corpus = [ dictionary.doc2bow(text) for text in train ]
  corpora.MmCorpus.serialize(result_dir + '/' + train_type + '_corpus.mm', corpus)
  return dictionary, corpus

# Loading dictionary and corpus directly
def load_dic_and_corpus(train_type, result_dir):
  dictionary = corpora.Dictionary.load(result_dir + '/' + train_type +'_dictionary.dictionary')
  corpus = corpora.MmCorpus(result_dir + '/' + train_type + '_corpus.mm')
  return dictionary, corpus

# 获取测试集语料
def get_test_corpus(dictionary, result_dir):
  test = load_text_pre(lda_test_tokenized_file)
  print("Finished reading processed test data!")
  corpus = [ dictionary.doc2bow(text) for text in test ]
  corpora.MmCorpus.serialize(result_dir + '/test_corpus.mm', corpus)
  return corpus

# 加载测试预料
def load_test_corpus(result_dir):
  return corpora.MmCorpus(result_dir + '/test_corpus.mm')

# 困惑度计算
def perplexity(ldamodel, testset, dictionary, size_dictionary, num_topics):
  """calculate the perplexity of a lda-model"""
  # dictionary : {7822:'deferment', 1841:'circuitry',19202:'fabianism'...]
  print ('the info of this ldamodel:')
  print ('num of testset: %s; size_dictionary: %s; num of topics: %s'%(len(testset), size_dictionary, num_topics))
  prep = 0.0
  prob_doc_sum = 0.0
  topic_word_list = [] # store the probablity of topic-word:[(u'business', 0.010020942661849608),(u'family', 0.0088027946271537413)...]
  for topic_id in range(num_topics):
    topic_word = ldamodel.show_topic(topic_id, size_dictionary)
    dic = {}
    for word, probability in topic_word:
        dic[word] = probability
    topic_word_list.append(dic)
  doc_topics_ist = [] #store the doc-topic tuples:[(0, 0.0006211180124223594),(1, 0.0006211180124223594),...]
  for doc in testset:
    doc_topics_ist.append(ldamodel.get_document_topics(doc, minimum_probability=0))
  testset_word_num = 0
  for i in range(len(testset)):
    prob_doc = 0.0 # the probablity of the doc
    doc = testset[i]
    doc_word_num = 0 # the num of words in the doc
    for word_id, num in doc:
      prob_word = 0.0 # the probablity of the word 
      doc_word_num += num
      word = dictionary[word_id]
      for topic_id in range(num_topics):
        # cal p(w) : p(w) = sumz(p(z)*p(w|z))
        prob_topic = doc_topics_ist[i][topic_id][1]
        prob_topic_word = topic_word_list[topic_id][word]
        prob_word += prob_topic*prob_topic_word
      prob_doc += math.log(prob_word) # p(d) = sum(log(p(w)))
    prob_doc_sum += prob_doc
    testset_word_num += doc_word_num
  prep = math.exp(-prob_doc_sum/testset_word_num) # perplexity = exp(-sum(p(d)/sum(Nd))
  return prep

# JS散度计算
def JS_shannon(p,q):
  M=(p+q)/2
  return 0.5*scipy.stats.entropy(p, M)+0.5*scipy.stats.entropy(q, M)

# 计算困惑度/JS散度
def cal_perplexity_var(ldamodel, dictionary, test_corpus, n_topic):
  size_dictionary = len(dictionary.keys())
  prep_score = perplexity(ldamodel, test_corpus, dictionary, size_dictionary, n_topic)

  term_topic_matrix = ldamodel.get_topics() #shape(num_topics, vocabulary_size)
  term_topic_mean = np.mean(term_topic_matrix, axis=0)
  js = float(0)
  for t in term_topic_matrix:
    js += math.sqrt(JS_shannon(t, term_topic_mean))
  var_score = js/n_topic

  prep_var = prep_score/var_score
  return prep_score, var_score, prep_var 


#在抽样数据集上找到最佳主题数
def find_best_topic_num(verify_type):
  dictionary, corpus = load_dic_and_corpus('sample', lda_sample_result_dir)
  test_corpus = load_test_corpus(lda_sample_result_dir)
  sample_train = load_text_pre(lda_sample_train_tokenized_file)
  print("Finished reading processed training data!")

  for n_topic in range(25, 100, 10):
    t0 = time()
    log = '%i topics: \n' % n_topic
    ldamodel = LdaModel.load(lda_sample_result_dir + "/lda_" + str(n_topic))
    if verify_type == 'coherence':
      # cm = CoherenceModel(model=ldamodel, corpus=corpus, coherence='u_mass')
      cm = CoherenceModel(model=ldamodel, texts=sample_train, dictionary=dictionary, coherence='c_v')
      coherence = cm.get_coherence()
      log += "the coherence is : %f \n" % coherence
      print(log)
      print("Finished cal_topic_coherence in %0.3fs \n" % (time() - t0))
    elif verify_type == 'prep_var':
      prep_score, var_score, prep_var = cal_perplexity_var(ldamodel, dictionary, test_corpus, n_topic)
      log += "the perp_score is : %f \n" % prep_score
      log += "the var_score is : %f \n" % var_score
      log += "the prep_var is : %f \n" % prep_var
      print(log)
      print("Finished cal_perplexity_var in %0.3fs \n" % (time() - t0))
  

def train_ldamodel(dictionary, corpus, n_topic, iter_num, result_dir):
  print("------------ %d Topics --------------" % n_topic)
  t0 = time()
  lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=n_topic, iterations=iter_num, alpha="auto", eta="auto")
  lda.save(result_dir + "/lda_" + str(n_topic))
  print("------------ Finished training lda model in %0.3fs --------------\n" % (time() - t0))


def train_sample_ldamodel():
  if not os.path.exists(lda_sample_result_dir):
    os.makedirs(lda_sample_result_dir)

  # 第一次加载, 获取.txt对应的研究内容，并作预处理，合并写入另一个txt文件，便于lda训练
  # get_lda_train_text(lda_sample_train_ids, lda_sample_train_tokenized_file)
  # get_lda_train_text(all_test_ids, lda_test_tokenized_file)

  # dictionary, corpus = get_dic_and_corpus('sample', lda_sample_result_dir, lda_sample_train_tokenized_file) #第一次加载
  dictionary, corpus = load_dic_and_corpus('sample', lda_sample_result_dir)

  print("Ready to train sample lda model...")
  for n_topic in range(25, 100, 10):
    train_ldamodel(dictionary, corpus, n_topic, 600, lda_sample_result_dir)


def train_final_ldamodel(n_topic):
  if not os.path.exists(lda_result_dir):
    os.makedirs(lda_result_dir)

  # 第一次加载, 获取.txt对应的研究内容，并作预处理，合并写入另一个txt文件，便于lda训练
  # get_lda_train_text(lda_train_ids, lda_train_tokenized_file)
  # get_lda_train_text(all_test_ids, lda_test_tokenized_file)

  # dictionary, corpus = get_dic_and_corpus('final', lda_result_dir, lda_train_tokenized_file) #第一次加载
  dictionary, corpus = load_dic_and_corpus('final', lda_result_dir)

  # print("Ready to train lda model...")
  train_ldamodel(dictionary, corpus, n_topic, 800, lda_result_dir)

  print("Starting to calculate perplexity_var score...")
  # test_corpus = get_test_corpus(dictionary, lda_result_dir) #第一次加载
  test_corpus = load_test_corpus(lda_result_dir)
  with open('sample_prep_var_log.txt', "w", encoding='utf-8') as writer:
    t0 = time()
    log = '%i topics: \n' % n_topic
    ldamodel = LdaModel.load(lda_result_dir + "/lda_" + str(n_topic))
    prep_score, var_score, prep_var = cal_perplexity_var(ldamodel, dictionary, test_corpus, n_topic)
    log += "the perp_score is : %f \n" % prep_score
    log += "the var_score is : %f \n" % var_score
    log += "the prep_var is : %f \n" % prep_var
    print(log)
    print("Finished cal_perplexity_var in %0.3fs \n" % (time() - t0))


if __name__ == '__main__':
  train_sample_ldamodel()
  find_best_topic_num('prep_var')
  train_final_ldamodel(65)