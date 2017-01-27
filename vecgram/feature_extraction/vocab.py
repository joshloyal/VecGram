import collections

from preshed.counter import PreshCounter
import spacy


def get_nlp(language='en'):
    return spacy.load(language, parser=False, tagger=False, entity=False)


def count_vocab(raw_documents, nlp=None, language='en'):
    nlp = get_nlp(language) if nlp is None else nlp
    for doc in nlp.pipe(raw_documents


