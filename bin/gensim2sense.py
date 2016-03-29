from sense2vec.vectors import VectorMap
from gensim.models import Word2Vec
import plac

@plac.annotations(
    gensim_model_path=("Location of gensim's .bin file"),
    out_dir=("Location of output directory"),
    min_count=("Min count", "option", "m", int),
)
def main(gensim_model_path, out_dir, min_count=None):
    """Convert a gensim.models.Word2Vec file to VectorMap format"""
    
    gensim_model = Word2Vec.load(gensim_model_path)
    vector_map = VectorMap(128)

    if min_count is None:
        min_count = gensim_model.min_count
        
    for string in gensim_model.vocab:
        vocab = gensim_model.vocab[string]
        freq, idx = vocab.count, vocab.index
        if freq < min_count:
            continue
        vector = gensim_model.syn0[idx]
        vector_map.borrow(string, freq, vector)
    
    vector_map.save(out_dir)

if __name__ == '__main__':
    plac.call(main)