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
def main(vectors, gpu_id=-1, n_neighbors=100, batch_size=1024, cutoff=0, start=0, end=None):
    if gpu_id == -1:
        xp = numpy
    else:
        import cupy as xp
        import cupy.cuda.device

        cupy.take_along_axis = take_along_axis
        cupy.put_along_axis = put_along_axis

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
        # Zero out the self-scores, to avoid returning self as a neighbor.
        self_indices = xp.arange(i, min(i + size, sims.shape[1])).reshape((1, -1))
        xp.put_along_axis(sims, self_indices, 0.0, axis=1)
        # Get the indices and scores for the top N most similar for each in the
        # batch. This is a bit complicated, to avoid sorting all of the scores
        # -- we only want the top N to be sorted (which we do later). For now,
        # we use argpartition to just get the cut point.
        neighbors = xp.argpartition(sims, -n, axis=1)[:, -n:]
        neighbor_sims = xp.partition(sims, -n, axis=1)[:, -n:]
        # Can't figure out how to do this without the loop.
        for j in range(min(end - i, size)):
            best_rows[i + j] = neighbors[j]
            scores[i + j] = neighbor_sims[j]
    # Sort in reverse order
    indices = xp.argsort(scores, axis=1)[:, ::-1]
    scores = xp.take_along_axis(scores, indices, axis=1)
    best_rows = xp.take_along_axis(best_rows, indices, axis=1)
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


# These functions are missing from cupy, but will be supported in cupy 7.
def take_along_axis(a, indices, axis):
    """Take values from the input array by matching 1d index and data slices.

    Args:
        a (cupy.ndarray): Array to extract elements.
        indices (cupy.ndarray): Indices to take along each 1d slice of ``a``.
        axis (int): The axis to take 1d slices along.

    Returns:
        cupy.ndarray: The indexed result.

    .. seealso:: :func:`numpy.take_along_axis`
    """
    import cupy

    if indices.dtype.kind not in ("i", "u"):
        raise IndexError("`indices` must be an integer array")

    if axis is None:
        a = a.ravel()
        axis = 0

    ndim = a.ndim

    if not (-ndim <= axis < ndim):
        raise _errors._AxisError("Axis overrun")

    axis %= a.ndim

    if ndim != indices.ndim:
        raise ValueError("`indices` and `a` must have the same number of dimensions")

    fancy_index = []
    for i, n in enumerate(a.shape):
        if i == axis:
            fancy_index.append(indices)
        else:
            ind_shape = (1,) * i + (-1,) + (1,) * (ndim - i - 1)
            fancy_index.append(cupy.arange(n).reshape(ind_shape))

    return a[fancy_index]


def put_along_axis(a, indices, value, axis):
    """Take values from the input array by matching 1d index and data slices.

    Args:
        a (cupy.ndarray): Array to extract elements.
        indices (cupy.ndarray): Indices to take along each 1d slice of ``a``.
        axis (int): The axis to take 1d slices along.

    Returns:
        cupy.ndarray: The indexed result.

    .. seealso:: :func:`numpy.take_along_axis`
    """
    import cupy

    if indices.dtype.kind not in ("i", "u"):
        raise IndexError("`indices` must be an integer array")

    if axis is None:
        a = a.ravel()
        axis = 0

    ndim = a.ndim

    if not (-ndim <= axis < ndim):
        raise _errors._AxisError("Axis overrun")

    axis %= a.ndim

    if ndim != indices.ndim:
        raise ValueError("`indices` and `a` must have the same number of dimensions")

    fancy_index = []
    for i, n in enumerate(a.shape):
        if i == axis:
            fancy_index.append(indices)
        else:
            ind_shape = (1,) * i + (-1,) + (1,) * (ndim - i - 1)
            fancy_index.append(cupy.arange(n).reshape(ind_shape))
    a[fancy_index] = value


if __name__ == "__main__":
    try:
        plac.call(main)
    except KeyboardInterrupt:
        msg.warn("Cancelled.")
