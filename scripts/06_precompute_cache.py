#!/usr/bin/env python
from typing import Optional
import tqdm
import numpy
import srsly
from wasabi import msg
from pathlib import Path
import typer


def main(
    # fmt: off
    vectors: str = typer.Argument(..., help="Path to sense2vec component directory"),
    gpu_id: int = typer.Option(-1, "--gpu-id", "-g", help="GPU device (-1 for CPU)"),
    n_neighbors: int = typer.Option(100, "--n-neighbors", "-n", help="Number of neighbors to cache"),
    batch_size: int = typer.Option(1024, "--batch-size", "-b", help="Batch size for to reduce memory usage"),
    cutoff: int = typer.Option(0, "--cutoff", "-c", help="Limit neighbors to this many earliest rows"),
    start: int = typer.Option(0, "--start", "-s", help="Index of vectors to start at"),
    end: Optional[int] = typer.Option(None, "--end", "-e", help="Index of vectors to stop at"),
    # fmt: on
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

        xp.take_along_axis = take_along_axis
        device = cupy.cuda.device.Device(gpu_id)
        cupy.cuda.get_cublas_handle()
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
    subset = vectors[:cutoff]
    best_rows = xp.zeros((end - start, n), dtype="i")
    scores = xp.zeros((end - start, n), dtype="f")
    for i in tqdm.tqdm(list(range(start, end, batch_size))):
        size = min(batch_size, end - i)
        batch = vectors[i : i + size]
        sims = xp.dot(batch, subset.T)
        # Set self-similarities to -inf, so that we don't return them.
        for j in range(size):
            if i + j < sims.shape[1]:
                sims[j, i + j] = -xp.inf
        # This used to use argpartition, to do a partial sort...But this ended
        # up being a ratsnest of terrible numpy crap. Just sorting the whole
        # list isn't really slower, and it's much simpler to read.
        ranks = xp.argsort(sims, axis=1)
        batch_rows = ranks[:, -n:]
        # Reverse
        batch_rows = batch_rows[:, ::-1]
        batch_scores = xp.take_along_axis(sims, batch_rows, axis=1)
        best_rows[i : i + size] = batch_rows
        scores[i : i + size] = batch_scores
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
        raise IndexError("Axis overrun")

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


if __name__ == "__main__":
    typer.run(main)
