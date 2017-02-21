from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import numpy as np
from preshed.counter import PreshCounter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
import spacy


from vecgram.parallel import get_n_jobs


def get_embeddings(vocab):
    max_rank = max(lex.rank for lex in vocab if lex.has_vector)
    vectors = np.ndarray((max_rank + 1, vocab.vectors_length), dtype=np.float32)
    for lex in vocab:
        if lex.has_vector:
            vectors[lex.rank] = lex.vector
    return vectors


def process_embeddings(embeddings, n_components, random_state=123):
    """Process the word embeddings by subtracting off the common
    mean and directions using the method in
    "All-but-the-Top: Simple but Effective Postprocessing for Word Representations"
    https://arxiv.org/pdf/1702.01417.pdf.
    """

    # subtract off the mean (no need to keep for test time
    # since the embedding is fixed)
    scaler = StandardScaler(with_std=False, copy=False)
    embeddings = scaler.fit_transform(embeddings)

    # perform a truncated svd
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    projections = svd.fit_transform(embeddings)
    embeddings -= np.dot(projections, svd.components_)

    return embeddings


def get_features(docs, n_docs, max_length=100):
    Xs = np.zeros((n_docs, max_length), dtype=np.int32)
    counts = PreshCounter()

    for i, doc in enumerate(docs):
        doc.count_by
        for j, token in enumerate(doc[:max_length]):
            if token.has_vector:
                Xs[i, j] = token.rank
                counts.inc(token.rank, 1)
            else:
                Xs[i, j] = 0
    return Xs, counts


class SpacyWordVectorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,
                 max_document_length=100,
                 language='en',
                 batch_size=10000,
                 post_process='pca',
                 n_components_threshold='auto',
                 n_jobs=1):
        self.max_document_length = max_document_length
        self.language = language
        self.batch_size = batch_size
        self.post_process = post_process
        self.n_components_threshold = n_components_threshold
        self.n_jobs = get_n_jobs(n_jobs)

        self.nlp_ = None
        self.embeddings_ = None
        self.vocabulary_ = None

    def fit(self, X, y=None):
        self.nlp_ = spacy.load(self.language, parser=False, tagger=False, entity=False)
        self.embeddings_ = get_embeddings(self.nlp_.vocab)

        if self.post_process == 'pca':
            if self.n_components_threshold == 'auto':
                k = int(self.embeddings_.shape[1] / 100.)
            else:
                k = self.n_components_threshold
            self.embeddings_ = process_embeddings(self.embeddings_, n_components=k)

        return self

    def transform(self, X):
        n_docs = len(X)
        doc_pipeline = self.nlp_.pipe(X, batch_size=self.batch_size, n_threads=self.n_jobs)
        Xs, counts = get_features(doc_pipeline, n_docs, max_length=self.max_document_length)
        self.vocabulary_ = counts
        return Xs



class AverageWordEmbedding(BaseEstimator, TransformerMixin):
    """Embed a sentence using an average of word vectors in the sentence."""
    def __init__(self,
                 language='en',
                 batch_size=10000,
                 n_jobs=1):
        self.language = language
        self.batch_size = batch_size
        self.n_jobs = get_n_jobs(n_jobs)

        self.nlp_ = None
        self.embeddings_ = None

    def fit(self, X, y=None):
        self.nlp_ = spacy.load(self.language, parser=False, tagger=False, entity=False)
        self.embeddings_ = get_embeddings(self.nlp_.vocab)
        return self

    def transform(self, X):
        n_docs = len(X)
        doc_pipeline = self.nlp_.pipe(X, batch_size=self.batch_size, n_threads=self.n_jobs)

        Xs = np.zeros((n_docs, self.nlp_.vocab.vectors_length), dtype=np.float32)
        for i, doc in enumerate(doc_pipeline):
            n_tokens = 0
            for j, token in enumerate(doc):
                if token.has_vector:
                    Xs[i, :] += token.vector
                    n_tokens += 1
            Xs[i, :] /= n_tokens

        return Xs


class SimpleSentenceEmbedding(SpacyWordVectorTransformer):
    def __init__(self,
                 alpha=1e-3,
                 max_document_length=100,
                 language='en',
                 batch_size=10000,
                 n_jobs=1):
        self.alpha = alpha
        super(SimpleSentenceEmbedding, self).__init__(
            max_document_length=max_document_length,
            language=language,
            batch_size=batch_size,
            n_jobs=n_jobs)

    def transform(self, X, y=None):
        Xs = super(SimpleSentenceEmbedding, self).transform(X)
        Es = np.zeros((Xs.shape[0], self.embeddings_.shape[1]), dtype=np.float32)
        for i in xrange(Xs.shape[0]):
            n_words = 0
            for j in xrange(Xs.shape[1]):
                word_id = Xs[i, j]
                if word_id > 0:
                    scale_factor = self.alpha / (self.alpha + self.vocabulary_[word_id])
                    Es[i, :] +=  scale_factor * self.embeddings_[word_id, :]
                    n_words += 1
            Es[i, :] /= n_words

        # compute first principle component of Es
        #pca = PCA(n_components=1, svd_solver='arpack', random_state=123).fit(Es)
        pca = TruncatedSVD(n_components=1, algorithm='arpack', random_state=123).fit(Es)
        Us = pca.components_[0, :]
        word_scale = np.dot(Us, Us.T)

        # set Es_i = Es_i - uu^T * Es_i
        Es -= word_scale * Es

        return Es
