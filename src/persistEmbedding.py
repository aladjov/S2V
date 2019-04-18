import numpy as np
from termcolor import colored

word2vector_dictonary = {}

def embed_using_pretrained_word2vector_dictionary(word_index, pretrained_vector, embedding_vector_dimensions):
    """
    Create embedding matrix using pretrained word dictionary.
    
    Parameters
    ----------
    word_index : dict
        Word index sorted by frequency (the most frequent word on top with the smallest index)
    pretrained_vector : str
        Path to the pretrained word to vector dictionary. The expected format is space separated word and its vector representation dimensions 
    embedding_vector_dimensions : int
        Number of dimension of pretrained vector representation of a word

    Returns
    -------
    Embedding
        Returns Embedding Matrix

    """
    load_pretrained_word2vector_dictionary(pretrained_vector)
    return create_embadding_matrix(word_index,embedding_vector_dimensions)

def load_pretrained_word2vector_dictionary(pretrained_vector):
    print(colored('Loading pretrained word vectors from',"cyan"),colored(pretrained_vector,"green"))
    word2vector_dictonary = {}
    with open(pretrained_vector) as pretrained_file:
    # is just a space-separated text file in the format:
    # word vec[0] vec[1] vec[2] ...
        for line in pretrained_file:
            line = line.replace(" \n","\n")
            values = line.split(" ")
            word = values[0]
            try:
                vec = np.asarray(values[1:], dtype='float32')
            except ValueError:
                print('Error in processing the vector for word',word)
            word2vector_dictonary[word] = vec
    print('Found %s word vectors.' % len(word2vector_dictonary))
    return word2vector_dictonary

def create_embadding_matrix(word_index, embedding_vector_dimensions):
    # prepare embedding matrix with dimensions: vocabulary_size x embedding_vector_dimensions
    vocabulary_size = len(word_index) + 1
    embedding_matrix = np.zeros((vocabulary_size, embedding_vector_dimensions))
    print(embedding_matrix.shape)
    for word, i in word_index.items():
        embedding_vector = word2vector_dictonary.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix



if __name__ == "__main__":
    cnt = 0
    word_index = {}
    with open('/home/vital/python/playgorund/nlp/assets/words_vector/glove.6B.300d_unk_vocab.txt','r') as vocab:
        for line in vocab:
            word_index[line.strip('\n')] = cnt
            cnt += 1
    embedding_matrix = embed_using_pretrained_word2vector_dictionary(word_index, '/home/vital/python/playgorund/nlp/assets/words_vector/glove.6B.300d_unk.txt', 300)
    np.save('/home/vital/python/playgorund/nlp/assets/words_vector/glove.6B.300d_unk_embed.npy', embedding_matrix)