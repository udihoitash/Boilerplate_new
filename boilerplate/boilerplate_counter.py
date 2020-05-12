import sys

sys.path.append("..")

from utils.connector import dataframe_to_postgres
from dataloader import get_data
from configs import config
from utils import preprocessor
import time
import logging
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer


def get_boiler_plate_words(vocab, X, upper_threshold, lower_threshold):
    total_docs = X.shape[0]
    document_frequency = X.sum(axis=0)
    vocab_doc_freq = dict(zip(vocab, document_frequency / total_docs))
    filtered_vocab_doc_freq = {k: v for k, v in vocab_doc_freq.items() if
                               (v >= lower_threshold and v <= upper_threshold)}
    if not filtered_vocab_doc_freq:
        raise Exception("No ngrams present with this threshold. Reduce the threshold.")
        exit(1)
    else:
        return filtered_vocab_doc_freq.keys()


def get_boiler_plate_word_count(boiler_phrases, row):
    sentences = row.split('.')
    boiler_count = 0
    for sentence in sentences:
        for phrase in boiler_phrases:
            if phrase in sentence:
                boiler_count += len(sentence.split())
                break
    return boiler_count


def main(preprocess, df_data, upper_threshold, lower_threshold):
    if preprocess:
        logging.info("Preprocessing the column...")
        df_data = preprocessor.preprocess_column(cfg, df_data, training_column, do_lemmatize=lemmatize,
                                                 no_stopwords=remove_stopwords)
    else:
        logging.info("You have opted not to preprocess the column.")
        df_data['processed_value'] = df_data[training_column]

    corpus = df_data[df_data['processed_value'].notnull()]
    corpus = corpus[corpus['processed_value'] != ""]

    doc_count = len(corpus)
    vectorizer = CountVectorizer(ngram_range=(int(ngrams[0]), int(ngrams[1])), max_features=max_features, binary=True)
    logging.info("Vectorizing and building ngrams...")
    X = vectorizer.fit_transform(corpus['processed_value'].values)

    # optimize to csr matrix if memory error
    X = X.toarray()
    vocab = vectorizer.get_feature_names()
    boiler_phrases = get_boiler_plate_words(vocab, X, upper_threshold, lower_threshold)
    logging.info("Identified boiler phrases are {}".format(boiler_phrases))

    tqdm.pandas()
    logging.info("Getting boiler word counts ...")
    corpus['boiler_word_count'] = corpus['processed_value'].progress_apply(
        lambda row: get_boiler_plate_word_count(boiler_phrases, row))
    logging.info("Calculating row lengths...")
    corpus['word_length'] = corpus['processed_value'].progress_apply(lambda row: row.split().__len__())
    corpus['boiler_perc'] = (corpus['boiler_word_count'] / corpus['word_length']) * 100

    # n_vocab = len(vocab)
    # total_freq = X.sum(axis=0).sum(axis=0)
    # logging.info("Total n_grams generated is {} ".format(X.shape))
    # df = pd.DataFrame(data=vocab, columns=['ngrams'])
    # logging.info("Now calculating different metrics...")
    # df['percentage_docs'] = (np.count_nonzero(X, axis=0) / doc_count) * 100
    # df['ngram_counts'] = X.sum(axis=0)
    # df['total_ngram_freq'] = total_freq
    # df['percentage_ngram'] = (df['ngram_counts'] / df['total_ngram_freq']) * 100
    return corpus


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    start = time.time()
    logging.info('Reading from configuration...')
    cfg = config.read()

    training_column = cfg.get('postgres', 'column')
    ngrams = (cfg.get('ngram', 'ngram_number')).split(',')
    max_features = cfg.getint('ngram', 'maximum_ngrams')
    preprocess = cfg.getboolean('preprocess', 'preprocess')
    remove_stopwords = cfg.getboolean('preprocess', 'remove_stopwords')
    lemmatize = cfg.getboolean('preprocess', 'lemmatize')
    split_column = cfg.get('ngram', 'by')
    upper_threshold = cfg.getfloat("boiler", "upper_threshold")
    lower_threshold = cfg.getfloat("boiler", "lower_threshold")

    output_table_name_generic = cfg.get('postgres', 'output_table')
    df_data = get_data.get_table(cfg, 'ALL', 0)

    logging.info('Reading the configurations...')
    logging.info("ngram_range: {} and max_ngrams: {} and preprocess: {}".format(ngrams, max_features, preprocess))

    if split_column:
        logging.info("Splitting the dataframe based on {}".format(split_column))

        dict_of_split_columns = {k: v for k, v in df_data.groupby(split_column)}

        for split_col, df_data in dict_of_split_columns.items():
            df = main(preprocess, df_data, upper_threshold, lower_threshold)
            logging.info('Saving the results in results/ folder as a csv file...')
            output_table_name = output_table_name_generic + "_" + str(split_col)
            df.to_csv("../results/" + output_table_name + ".csv", index=False)

            logging.info('Now copying the results into the database...')
            dataframe_to_postgres(df, output_table_name, "False")
            logging.info('TASK COMPLETED in {} minutes!'.format((time.time() - start) / 60.0))

    else:

        df = main(preprocess, df_data, upper_threshold, lower_threshold)

        logging.info('Saving the results in results/ folder as a csv file...')
        df.to_csv("../results/" + output_table_name_generic + ".csv", index=False)

        logging.info('Now copying the results into the database...')
        dataframe_to_postgres(df, output_table_name_generic, "False")
        logging.info('TASK COMPLETED in {} minutes!'.format((time.time() - start) / 60.0))
