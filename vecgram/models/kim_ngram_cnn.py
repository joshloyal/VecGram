from keras.layers import Input, Dense, Dropout
from keras.layers import Flatten, Reshape, merge
from keras.layers import Conv2D, MaxPooling2D, Embedding
from keras.models import Model


def ngram_cnn(n_classes,
              n_vocab,
              max_length,
              embedding_size,
              ngram_filters=[2, 3, 4, 5],
              n_feature_maps=128,
              dropout=0.5,
              n_hidden=15,
              init_embeddings=None,
              train_embeddings=False):
    """A single-layer convolutional network using different n-gram filters.

    Parameters
    ----------
    n_vocab: int
        Number of words in the corpus vocabulary.
    max_length: int
        Maximum sentence length in the corpus.
    embedding_size: int
        Size of the dense embedding layer.
    ngram_filters: iterable
        N-gram filter sizes for the convolutional layers.
    n_feature_maps: int
        The number of feature maps used for each filter.
    dropout: float
        Dropout probability for the dropout layer after the conv layers.
    n_hidden: int
        Number of hidden units used in the fully-connected layer
    init_embeddingss: np.array or None
        Array to use to set the initial word embeddings

    References
    ----------
    A Sensitivity Analysis of Convolutional Neural Networks for Sentence Classification.
         Ye Zhang, Byron C. Wallace. <http://arxiv.org/pdf/1510.03820v3.pdf>
    Convolutional Neural Networks for Sentence Classification.
         Yoon Kim. <http://arxiv.org/pdf/1408.5882v2.pdf>
    """
    sequence_input = Input(shape=(max_length,), dtype='int32')
    embedding_layer = Embedding(
            n_vocab,
            embedding_size,
            input_length=max_length,
            weights=[init_embeddings],
            trainable=train_embeddings)(sequence_input)
    conv_input = Reshape((1, max_length, embedding_size))(embedding_layer)

    # convolutional n-grams
    def ngram_conv_module(x, n_gram):
        x = Conv2D(n_feature_maps, n_gram, embedding_size, activation='relu')(conv_input)
        x = MaxPooling2D(pool_size=(max_length - n_gram + 1, 1))(x)
        x = Flatten()(x)
        return x
    ngram_convs = merge([ngram_conv_module(conv_input, n_gram) for n_gram in ngram_filters], mode='concat')

    # fully-connected layer
    x = Dropout(dropout)(ngram_convs)
    # x = Dense(100, activation='relu')(x)
    preds = Dense(n_classes, activation='softmax')(x)

    # model
    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    return model
