from __future__ import unicode_literals

from vecgram.feature_extraction import (
    SpacyWordVectorTransformer,
    AverageWordEmbedding,
    SimpleSentenceEmbedding)


def test_smoke_base():
    docs = ['hi my name is josh', 'what is your name']

    wv = SpacyWordVectorTransformer()
    print(wv.fit_transform(docs))


def test_smoke_average():
    docs = ['hi my name is josh', 'what is your name']
    wv = AverageWordEmbedding()
    print(wv.fit_transform(docs))


def test_simple_sentence():
    docs = ['hi my name is josh', 'what is your name']
    wv = SimpleSentenceEmbedding()
    print(wv.fit_transform(docs))
