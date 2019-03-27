import sys
import os
import struct
import collections
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.core.example import example_pb2
from gensim import corpora
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from sklearn.decomposition import PCA

END_TOKENS = ['。', '；', ';', '！', '!', '...', '......', '？', '?']
SERIAL_TOKENS = ['①', '②', '③', '④', '⑤', '⑥', '⑦', '⑧', '⑨', '⑩']

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

all_train_ids = './id_lists/all_train_ids.txt'
all_test_ids = './id_lists/all_test_ids.txt'
all_eval_ids = './id_lists/all_eval_ids.txt'

content_tokenized_dir = "./content_tokenized"
target_tokenized_dir = "./target_tokenized"

num_projects = 202408

VOCAB_SIZE = 200000
CHUNK_SIZE = 1000 # num examples per chunk, for the chunked data

def read_text_file(text_file):
  lines = []
  with open(text_file, "r", encoding="utf-8") as f:
    for line in f:
      if line.strip() != '': lines.append(line.strip())
  return lines

stopwords = read_text_file('../Chinese_stopwords_1893.txt')

def chunk_file(set_name, finished_files_dir, chunks_dir):
  in_file = finished_files_dir + '/%s.bin' % set_name
  reader = open(in_file, "rb")
  chunk = 0
  finished = False
  while not finished:
    chunk_fname = os.path.join(chunks_dir, '%s_%03d.bin' % (set_name, chunk)) # new chunk
    with open(chunk_fname, 'wb') as writer:
      for _ in range(CHUNK_SIZE):
        len_bytes = reader.read(8)
        if not len_bytes:
          finished = True
          break
        str_len = struct.unpack('q', len_bytes)[0]
        example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
        writer.write(struct.pack('q', str_len))
        writer.write(struct.pack('%ds' % str_len, example_str))
      chunk += 1


def chunk_all(finished_files_dir, chunks_dir):
  # Make a dir to hold the chunks
  if not os.path.isdir(chunks_dir):
    os.mkdir(chunks_dir)
  # Chunk the data
  for set_name in ['train', 'val', 'test']:
    print("Splitting %s data into chunks..." % set_name)
    chunk_file(set_name, finished_files_dir, chunks_dir)
  print("Saved chunked data in %s" % chunks_dir)


def divide_sents(txt_lines):
  word_list = ' '.join(txt_lines).split()
  word_num = len(word_list)
  sents = []
  last_idx = 0

  for idx,w in enumerate(word_list):
    if w in END_TOKENS:
      _sent = word_list[last_idx:idx+1]
      if len(_sent) > 3:
        sents.append(_sent)
      last_idx = idx + 1
    elif w in SERIAL_TOKENS:
      _sent = word_list[last_idx:idx]
      if len(_sent) > 3:
        sents.append(_sent)
      last_idx = idx + 1

  if last_idx < word_num:
    sents.append(word_list[last_idx:word_num])
  return sents


def get_art_abs(content_file, target_file):
  article_lines = read_text_file(content_file)
  article = ' '.join(article_lines)

  abstract_lines = read_text_file(target_file)
  abstract_sents = divide_sents(abstract_lines)
  # Make abstract into a signle string, putting <s> and </s> tags around the sentences
  abstract = ' '.join(["%s %s %s" % (SENTENCE_START, ' '.join(sent), SENTENCE_END) for sent in abstract_sents])

  return article, abstract

# 去除停用词，已分好词，空格隔开
def txt_processing(text):
  # 分词
  wordLst = text.split(' ')
  # 去除停用词
  filtered = [w for w in wordLst if w not in stopwords]
  return ' '.join(filtered)


def write_to_bin(id_file, out_file, args, makevocab=False):
  """Reads the tokenized .txt files corresponding to the ids listed in the id_file and writes them to a out_file."""
  print("Making bin file for ids listed in %s..." % id_file)
  id_list = read_text_file(id_file)
  txt_fnames = [s+".txt" for s in id_list]
  num_txts = len(txt_fnames)

  if makevocab:
    vocab_counter = collections.Counter()

  if args.isTopicAware:
    ldamodel = LdaModel.load(args.lda_model_path)
    dictionary = Dictionary.load(args.dictionary_path)

  with open(out_file, 'wb') as writer:
    for idx,s in enumerate(txt_fnames):
      if idx % 1000 == 0:
        print("Writing txt %i of %i; %.2f percent done" % (idx, num_txts, float(idx)*100.0/float(num_txts)))

      # Look in the tokenized txt dirs to find the content.txt file corresponding to this id
      if os.path.isfile(os.path.join(content_tokenized_dir, s)):
        content_file = os.path.join(content_tokenized_dir, s)
      else:
        print("Error: Couldn't find tokenized txt file %s in either tokenized txt directories %s. Was there an error during tokenization?" % (s, content_tokenized_dir))
        # Check again if tokenized txts directories contain correct number of files
        print("Checking that the tokenized txts directories %s contain correct number of files..." % (content_tokenized_dir))
        check_num_stories(content_tokenized_dir, num_projects)
        raise Exception("Tokenized txts directories %s contain correct number of files but content txt file %s found in neither." % (content_tokenized_dir, s))

      # Look in the tokenized txt dirs to find the target.txt file corresponding to this id
      if os.path.isfile(os.path.join(target_tokenized_dir, s)):
        target_file = os.path.join(target_tokenized_dir, s)
      else:
        print("Error: Couldn't find tokenized txt file %s in either tokenized txt directories %s. Was there an error during tokenization?" % (s, target_tokenized_dir))
        # Check again if tokenized txts directories contain correct number of files
        print("Checking that the tokenized txts directories %s contain correct number of files..." % (target_tokenized_dir))
        check_num_stories(target_tokenized_dir, num_projects)
        raise Exception("Tokenized txts directories %s contain correct number of files but target txt file %s found in neither." % (target_tokenized_dir, s))

      # Get the strings to write to .bin file
      article, abstract = get_art_abs(content_file, target_file)

      # If args.isTopicAware is True, Get the topic distribution
      if args.isTopicAware:
        top_topic, dist_of_word2topic = get_dist_from_lda(dictionary, article, ldamodel, args)
        topic_distribution = '/'.join(top_topic)
        word2topic = ' '.join(dist_of_word2topic)

      # Write to tf.Example
      tf_example = example_pb2.Example()
      tf_example.features.feature['article'].bytes_list.value.extend([article.encode()])
      tf_example.features.feature['abstract'].bytes_list.value.extend([abstract.encode()])
      if args.isTopicAware:
        tf_example.features.feature['topicDistribution'].bytes_list.value.extend([topic_distribution.encode()])
        tf_example.features.feature['word2topic'].bytes_list.value.extend([word2topic.encode()])
      tf_example_str = tf_example.SerializeToString()
      str_len = len(tf_example_str)
      writer.write(struct.pack('q', str_len))
      writer.write(struct.pack('%ds' % str_len, tf_example_str))

      # Write the vocab to file, if applicable
      if makevocab:
        art_tokens = article.split(' ')
        abs_tokens = abstract.split(' ')
        abs_tokens = [t for t in abs_tokens if t not in [SENTENCE_START, SENTENCE_END]] # remove these tags from vocab
        tokens = art_tokens + abs_tokens
        tokens = [t.strip() for t in tokens] # strip
        tokens = [t for t in tokens if t!=""] # remove empty
        vocab_counter.update(tokens)

  print("Finished writing file %s\n" % out_file)

  # write vocab to file
  if makevocab:
    print("Writing vocab file...")
    with open(os.path.join(finished_files_dir, "vocab"), 'w', encoding="utf-8") as writer:
      for word, count in vocab_counter.most_common(VOCAB_SIZE):
        writer.write(word + ' ' + str(count) + '\n')
    print("Finished writing vocab file")


def get_dist_from_lda(dictionary, article, ldamodel, args):
  """ Get the topic distribution for the given document. 
  Args:
    dictionary
    article: String, original document
    ldamodel
    agrs: Some pre-set parameters
  Returns:
    top_topic: Array, the topics most relevant to the document and the corresponding probability, 
    ['topicId-prob', 'topicId-prob', ...]
    low_dim_word2topic: Array, each element in the list is a pair of a topic id and phi value between this word and each topic
    ['phi/phi/...', 'phi/phi/...', ]
  """
  n_topic = args.n_topic
  beta = args.beta
  isReduceDimensionality = args.dim_reduce
  n_components = args.pca_n_components

  processed_article = txt_processing(article)
  doc2bow = dictionary.doc2bow(processed_article.split())
  doc_topic = ldamodel.get_document_topics(doc2bow, minimum_probability=0, minimum_phi_value=None, per_word_topics=True)
  # Topic distribution for the whole document.
  topic_distribution = doc_topic[0]
  # Most probable topics per word. Each element in the list is a pair of a word’s id, and a list of topics sorted by their relevance to this word. 
  word_topics = doc_topic[2]
  word_topics_pair = {} #Change to dictionary
  for item in word_topics:
    word_topics_pair[item[0]] = item[1]
  topic_word_ids = word_topics_pair.keys()

  # Get the most relative topic
  topic_distribution_sorted = sorted(topic_distribution, key = lambda k:k[1], reverse = True)
  most_relative_topic = []
  sum_prob = float(0)
  for topic in topic_distribution_sorted:
    most_relative_topic.append(topic)
    sum_prob += topic[1]
    if sum_prob > beta:
      break
  top_topic_ids = [t[0] for t in most_relative_topic]
  top_topic = [str(t[0])+'-'+str(t[1]) for t in most_relative_topic]

  # Get word_to_topic_relevance_value
  dist_of_word2topic = []
  word2topic_str = []
  article_ids = dictionary.doc2idx(article.split(' '))
  for idx in article_ids:
    if idx in topic_word_ids:
      item = word_topics_pair[idx]
      topic_and_phi = [float(0)]*n_topic
      for t in item:
        topic_id = t[0]
        if topic_id in top_topic_ids:
          topic_and_phi[topic_id] = t[1]
      if isReduceDimensionality:
        dist_of_word2topic.append(topic_and_phi)
      else:
        topic_and_phi = [str(val) for val in topic_and_phi]
        word2topic_str.append('/'.join(topic_and_phi))
    else:
      word2topic_str.append('[stopwords]')

  if len(word2topic_str) != len(article_ids):
    print('length error!')

  if isReduceDimensionality:
    # dimensionality reduction using PCA (sklearn)
    pca = PCA(n_components=n_components, copy=False, svd_solver="randomized")
    dim_red_res = pca.fit_transform(dist_of_word2topic)
    pad_len = n_components - len(dim_red_res[0])
    for idx in article_ids:
      if idx in topic_word_ids:
        dist = dim_red_res.pop()
        if len(dist) < n_components:
          dist = np.hstack((dist, np.zeros(pad_len)))
        dist = [str(v) for v in dist]
        word2topic_str.append('/'.join(dist))
      else:
        word2topic_str.append('[stopwords]')

  return top_topic, word2topic_str


def check_num_stories(txt_dir, num_expected):
  num_txt = len(os.listdir(txt_dir))
  if num_txt != num_expected:
    raise Exception("txt directory %s contains %i files but should contain %i" % (txt_dir, num_txt, num_expected))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="make datafiles", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--finished_files_dir', '-fd', type=str, default="../model/finished_files_test", help='the path of finished files dir')
  parser.add_argument('--isTopicAware', '-ta', type=bool, default=True, help='whether to add topic distribution info to the story')
  parser.add_argument('--lda_model_path', '-l', type=str, default="./lda_result/lda_65", help='the path of lda model')
  parser.add_argument('--n_topic', '-nt', type=int, default=65, help='Number of topics of lda model.')
  parser.add_argument('--beta', '-b', type=float, default=0.9, help='We only need topics which the sum of the probabilities is greater than this threshold.')
  parser.add_argument('--dictionary_path', '-dic', type=str, default="./lda_result/final_dictionary.dictionary", help='the path of dictionary')
  parser.add_argument('--dim_reduce', '-dr', type=bool, default=False, help='whether to reduce the dimensionality by PCA')
  parser.add_argument('--pca_n_components', '-np', type=int, default=30, help='Number of components to keep using PCA of sklearn.')
  args = parser.parse_args()

  print('lda model is %s\n' % (args.lda_model_path))

  # Check the content and target directories contain the correct number of .txt files
  check_num_stories(content_tokenized_dir, num_projects)
  check_num_stories(target_tokenized_dir, num_projects)

  # Create some new directories
  finished_files_dir = args.finished_files_dir
  if not os.path.exists(finished_files_dir): os.makedirs(finished_files_dir)

  # Read the tokenized stories, do a little postprocessing then write to bin files
  write_to_bin(all_test_ids, os.path.join(finished_files_dir, "test.bin"), args)
  write_to_bin(all_eval_ids, os.path.join(finished_files_dir, "val.bin"), args)
  write_to_bin(all_train_ids, os.path.join(finished_files_dir, "train.bin"), args, makevocab=True)

  # Chunk the data. This splits each of train.bin, val.bin and test.bin into smaller chunks, each containing e.g. 1000 examples, and saves them in finished_files/chunks
  chunks_dir = os.path.join(finished_files_dir, "chunked")
  chunk_all(finished_files_dir, chunks_dir)
