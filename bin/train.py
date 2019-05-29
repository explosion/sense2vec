from gensim.models import Word2Vec
from gensim.models.word2vec import PathLineSentences
from sense2vec.vectors import VectorMap
import plac


@plac.annotations(
    in_dir=("Location of input directory", "positional", None, str),
    out_file=("Location of output file", "positional", None, str),
    n_workers=("Number of workers", "option", "n", int),
    size=("Dimension of the word vectors", "option", "d", int),
    window=("Context window size", "option", "w", int),
    min_count=("Min count", "option", "m", int),
    negative=("Number of negative samples", "option", "g", int),
    nr_iter=("Number of iterations", "option", "i", int),)
def train(in_dir, out_file, negative=5, n_workers=4, window=5, size=128,
          min_count=10, nr_iter=2):
    w2v_model = Word2Vec(size=size, window=window, min_count=min_count,
                         workers=workers, sample=1e-5, negative=negative,
                        iter=epochs)
    sentences = PathLineSentences(in_dir)
    print("Building the vocabulary...")
    w2v_model.build_vocab(sentences)
    print("Training the model...")
    w2v_model.train(sentences, total_examples=w2v_model.corpus_count,
                    epochs=w2v_model.iter)
    print("Creating the sense2vec model...")
    vector_map = VectorMap(size)
    for string in w2v_model.wv.vocab:
        vocab = w2v_model.wv.vocab[string]
        freq, idx = vocab.count, vocab.index
        if freq < min_count:
            continue
        vector = w2v_model.wv.vectors[idx]
        vector_map.borrow(string, freq, vector)
    print("Saving the model...")
    vector_map.save(out_file)
    print("Saved model to file: ", out_file)


if __name__ == '__main__':
    plac.call(main)
