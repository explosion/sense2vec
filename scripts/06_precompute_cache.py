import plac
import tqdm
import numpy
import srsly
from wasabi import msg
from pathlib import Path


@plac.annotations(
    vectors=("Path to sense2vec component directory", "positional", None, str),
    gpu_id=("GPU device (-1 for CPU)", "option", "g", int),
    n_neighbors=("Number of neighbors to cache", "option", "n", int),
    batch_size=("Batch size for to reduce memory usage.", "option", "b", int),
    cutoff=("Limit neighbors to this many earliest rows", "option", "c", int,),
)
def main(vectors, gpu_id=-1, n_neighbors=100, batch_size=1024, cutoff=0):
    """
    Step 6: Precompute nearest-neighbor queries (optional)

    Precompute nearest-neighbor queries for every entry in the vocab to make
    Sense2Vec.most_similar faster. The --cutoff option lets you define the
    number of earliest rows to limit the neighbors to. For instance, if cutoff
    is 100000, no word will have a nearest neighbor outside of the top 100k
    vectors.
    """
    if gpu_id == -1:
        xp = numpy
    else:
        import cupy as xp
        import cupy.cuda.device

        device = cupy.cuda.device.Device(gpu_id)
        device.use()

    vectors_dir = Path(vectors)
    vectors_file = vectors_dir / "vectors"
    if not vectors_dir.is_dir() or not vectors_file.exists():
        err = "Are you passing in the exported sense2vec directory containing a vectors file?"
        msg.fail(f"Can't load vectors from {vectors}", err, exits=1)
    with msg.loading(f"Loading vectors from {vectors}"):
        vectors = xp.load(str(vectors_file))
    msg.good(f"Loaded {vectors.shape[0]:,} vectors with dimension {vectors.shape[1]}")
    norms = xp.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    # Normalize to unit norm
    vectors /= norms
    if cutoff < 1:
        cutoff = vectors.shape[0]
    msg.good(f"Normalized (mean {norms.mean():,.2f}, variance {norms.var():,.2f})")
    msg.info(f"Finding {n_neighbors:,} neighbors among {cutoff:,} most frequent")
    best_rows = xp.zeros((vectors.shape[0], n_neighbors), dtype="i")
    scores = xp.zeros((vectors.shape[0], n_neighbors), dtype="f")
    # Pre-allocate this array, so we can use it each time.
    subset = xp.ascontiguousarray(vectors[:cutoff])
    sims = xp.zeros((batch_size, cutoff), dtype="f")
    for i in tqdm.tqdm(list(range(0, vectors.shape[0], batch_size))):
        batch = vectors[i : i + batch_size]
        # batch   e.g. (1024, 300)
        # vectors e.g. (10000, 300)
        # sims    e.g. (1024, 10000)
        if batch.shape[0] == sims.shape[0]:
            xp.dot(batch, subset.T, out=sims)
        else:
            # In the last batch we'll have a different size.
            sims = xp.dot(batch, subset.T)
        size = sims.shape[0]
        batch_indices = xp.argpartition(sims, -n_neighbors, axis=1)[:, -n_neighbors:]
        # God, I hate numpy. There must be a way to write this without the loop.
        batch_scores = xp.zeros((size, n_neighbors), dtype="f")
        for i in range(batch_indices.shape[0]):
            batch_scores[i] = sims[i, batch_indices[i]]
        best_rows[i : i + size] = batch_indices
        scores[i : i + size] = batch_scores
    output_file = vectors_dir / "cache"
    with msg.loading("Saving output..."):
        srsly.write_msgpack(output_file, {"indices": best_rows, "scores": scores})
    msg.good(f"Saved cache to {output_file}")


if __name__ == "__main__":
    try:
        plac.call(main)
    except KeyboardInterrupt:
        msg.warn("Cancelled.")
