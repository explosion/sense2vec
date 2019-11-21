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
    start=("Index of vectors to start at.", "option", "s", int),
    end=("Index of vectors to stop at.", "option", "e", int),
)
def main(
    vectors, gpu_id=-1, n_neighbors=100, batch_size=1024, cutoff=0, start=0, end=None
):
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
    if end is None:
        end = vectors.shape[0]
    mean = float(norms.mean())
    var = float(norms.var())
    msg.good(f"Normalized (mean {mean:,.2f}, variance {var:,.2f})")
    msg.info(f"Finding {n_neighbors:,} neighbors among {cutoff:,} most frequent")
    n = min(n_neighbors, vectors.shape[0])
    best_rows = xp.zeros((end - start, n), dtype="i")
    scores = xp.zeros((end - start, n), dtype="f")
    # Pre-allocate this array, so we can use it each time.
    subset = xp.ascontiguousarray(vectors[:cutoff])
    sims = xp.zeros((batch_size, cutoff), dtype="f")
    indices = xp.arange(cutoff).reshape((-1, 1))
    for i in tqdm.tqdm(list(range(start, end, batch_size))):
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
        # Get the indices and scores for the top N most similar for each in the
        # batch. This is a bit complicated, to avoid sorting all of the scores
        # -- we only want the top N to be sorted (which we do later). For now,
        # we use argpartition to just get the cut point.
        neighbors = xp.argpartition(sims, -n, axis=1)[:, -n:]
        neighbor_sims = xp.partition(sims, -n, axis=1)[:, -n:]
        # Can't figure out how to do this without the loop.
        for j in range(min(end - i, size)):
            # Sort in reverse order
            indices = xp.argsort(neighbor_sims[j], axis=-1)[::-1]
            best_rows[i + j] = xp.take(neighbors[j], indices)
            scores[i + j] = xp.take(neighbor_sims[j], indices)
    msg.info("Saving output")
    if not isinstance(best_rows, numpy.ndarray):
        best_rows = best_rows.get()
    if not isinstance(scores, numpy.ndarray):
        scores = scores.get()
    output = {
        "indices": best_rows,
        "scores": scores.astype("float16"),
        "start": start,
        "end": end,
        "cutoff": cutoff,
    }
    output_file = vectors_dir / "cache"
    with msg.loading("Saving output..."):
        srsly.write_msgpack(output_file, output)
    msg.good(f"Saved cache to {output_file}")


if __name__ == "__main__":
    try:
        plac.call(main)
    except KeyboardInterrupt:
        msg.warn("Cancelled.")
