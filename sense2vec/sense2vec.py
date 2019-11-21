from typing import Tuple, List, Union, Sequence, Dict, Callable, Any
from pathlib import Path
from spacy.vectors import Vectors
from spacy.strings import StringStore
import numpy
import srsly

from .util import registry, SimpleFrozenDict


class Sense2Vec(object):
    def __init__(
        self,
        shape: tuple = (1000, 128),
        strings: StringStore = None,
        senses: List[str] = [],
        vectors_name: str = "sense2vec",
        overrides: Dict[str, str] = SimpleFrozenDict(),
    ):
        """Initialize the Sense2Vec object.

        shape (tuple): The vector shape.
        strings (StringStore): Optional string store. Will be created if it
            doesn't exist.
        senses (list): Optional list of all available senses. Used in methods
            that generate the best sense or other senses.
        vectors_name (unicode): Optional name to assign to the Vectors object.
        overrides (dict): Optional custom functions to use, mapped to names
            registered via the registry, e.g. {"make_key": "custom_make_key"}.
        RETURNS (Sense2Vec): The newly constructed object.
        """
        self.vectors = Vectors(shape=shape, name=vectors_name)
        self.strings = StringStore() if strings is None else strings
        self.freqs: Dict[int, int] = {}
        self.cache = None
        self.cfg: Dict[str, Any] = {
            "senses": senses,
            "make_key": "default",
            "split_key": "default",
        }
        self.cfg.update(overrides)

    @property
    def senses(self) -> Sequence[str]:
        """RETURNS (list): The available senses."""
        return self.cfg.get("senses", [])

    @property
    def frequencies(self) -> List[Tuple[str, int]]:
        """RETURNS (list): The (key, freq) tuples by frequency, descending."""
        freqs = [(self.strings[k], s) for k, s in self.freqs.items() if s is not None]
        return sorted(freqs, key=lambda item: item[1], reverse=True)

    def __len__(self) -> int:
        """RETURNS (int): The number of rows in the vectors table."""
        return len(self.vectors)

    def __contains__(self, key: Union[str, int]) -> bool:
        """Check if a key is in the vectors table.

        key (unicode / int): The key to look up.
        RETURNS (bool): Whether the key is in the table.
        """
        key = self.ensure_int_key(key)
        return key in self.vectors

    def __getitem__(self, key: Union[str, int]) -> Union[numpy.ndarray, None]:
        """Retrieve a vector for a given key. Returns None if the key is not
        in the table.

        key (unicode / int): The key to look up.
        RETURNS (numpy.ndarray): The vector.
        """
        key = self.ensure_int_key(key)
        if key in self.vectors:
            return self.vectors[key]
        return None

    def __setitem__(self, key: Union[str, int], vector: numpy.ndarray):
        """Set a vector for a given key. Will raise an error if the key
        doesn't exist.

        key (unicode / int): The key.
        vector (numpy.ndarray): The vector to set.
        """
        key = self.ensure_int_key(key)
        if key not in self.vectors:
            raise ValueError(f"Can't find key {key} in table")
        self.vectors[key] = vector

    def __iter__(self):
        """YIELDS (tuple): String key and vector pairs in the table."""
        yield from self.items()

    def items(self):
        """YIELDS (tuple): String key and vector pairs in the table."""
        for key, value in self.vectors.items():
            yield self.strings[key], value

    def keys(self):
        """YIELDS (unicode): The string keys in the table."""
        for key in self.vectors.keys():
            yield self.strings[key]

    def values(self):
        """YIELDS (numpy.ndarray): The vectors in the table."""
        yield from self.vectors.values()

    @property
    def make_key(self) -> Callable:
        """Get the function to make keys."""
        return registry.make_key.get(self.cfg["make_key"])

    @property
    def split_key(self) -> Callable:
        """Get the function to split keys."""
        return registry.split_key.get(self.cfg["split_key"])

    def add(self, key: Union[str, int], vector: numpy.ndarray, freq: int = None):
        """Add a new vector to the table.

        key (unicode / int): The key to add.
        vector (numpy.ndarray): The vector to add.
        freq (int): Optional frequency count.
        """
        if not isinstance(key, int):
            key = self.strings.add(key)
        self.vectors.add(key, vector=vector)
        if freq is not None:
            self.set_freq(key, freq)

    def get_freq(self, key: Union[str, int], default=None) -> Union[int, None]:
        """Get the frequency count for a given key.

        key (unicode / int): They key to look up.
        default: Default value to return if no frequency is found.
        RETURNS (int): The frequency count.
        """
        key = self.ensure_int_key(key)
        return self.freqs.get(key, default)

    def set_freq(self, key: Union[str, int], freq: int):
        """Set a frequency count for a given key.

        key (unicode / int): The key to set the count for.
        freq (int): The frequency count.
        """
        if not isinstance(freq, int):
            raise ValueError(f"Invalid frequency count: {repr(freq)} for '{key}'")
        key = self.ensure_int_key(key)
        self.freqs[key] = freq

    def ensure_int_key(self, key: Union[str, int]) -> int:
        """Ensure that a key is an int by looking it up in the string store.

        key (unicode / int): The key.
        RETURNS (int): The integer key.
        """
        return key if isinstance(key, int) else self.strings.add(key)

    def similarity(
        self,
        keys_a: Union[Sequence[Union[str, int]], str, int],
        keys_b: Union[Sequence[Union[str, int]], str, int],
    ) -> float:
        """Make a semantic similarity estimate of two keys or two sets of keys.
        The default estimate is cosine similarity using an average of vectors.

        keys_a (unicode / int / iterable): The string or integer key(s).
        keys_b (unicode / int / iterable): The other string or integer key(s).
        RETURNS (float): The similarity score.
        """
        if isinstance(keys_a, (str, int)):
            keys_a = [keys_a]
        if isinstance(keys_b, (str, int)):
            keys_b = [keys_b]
        average_a = numpy.vstack([self[key] for key in keys_a]).mean(axis=0)
        average_b = numpy.vstack([self[key] for key in keys_b]).mean(axis=0)
        if average_a.all() == 0 or average_b.all() == 0:
            return 0.0
        norm_a = numpy.linalg.norm(average_a)
        norm_b = numpy.linalg.norm(average_b)
        if norm_a == norm_b:
            return 1.0
        return numpy.dot(average_a, average_b) / (norm_a * norm_b)

    def most_similar(
        self,
        keys: Union[Sequence[Union[str, int]], str, int],
        n: int = 10,
        batch_size: int = 16,
    ) -> List[Tuple[str, float]]:
        """Get the most similar entries in the table. If more than one key is
        provided, the average of the vectors is used.

        keys (unicode / int / iterable): The string or integer key(s) to compare to.
        n (int): The number of similar keys to return.
        batch_size (int): The batch size to use.
        RETURNS (list): The (key, score) tuples of the most similar vectors.
        """
        if isinstance(keys, (str, int)):
            keys = [keys]
        # Always ask for more because we'll always get the keys themselves
        n_similar = n + len(keys)
        for key in keys:
            if key not in self:
                raise ValueError(f"Can't find key {key} in table")
        if len(self.vectors) < n_similar:
            n_similar = len(self.vectors)
        if self.cache:
            indices = self.cache.get("indices", [])
            scores = self.cache.get("scores", [])
            if len(indices) >= n_similar:
                key_row = self.vectors.find(key=key)
                sim_keys = self.vectors.find(rows=indices[key_row][:n_similar])
                sim_scores = scores[key_row][:n_similar]
                result = [(self.strings[k], s) for k, s in zip(sim_keys, sim_scores)]
                return [(key, score) for key, score in result if key not in keys]
        vecs = numpy.vstack([self[key] for key in keys])
        average = vecs.mean(axis=0, keepdims=True)
        result_keys, _, scores = self.vectors.most_similar(
            average, n=n_similar, batch_size=batch_size
        )
        result = list(zip(result_keys.flatten(), scores.flatten()))
        result = [(self.strings[key], score) for key, score in result if key]
        result = [(key, score) for key, score in result if key not in keys]
        return result

    def get_other_senses(
        self, key: Union[str, int], ignore_case: bool = True
    ) -> List[str]:
        """Find other entries for the same word with a different sense, e.g.
        "duck|VERB" for "duck|NOUN".

        key (unicode / int): The key to check.
        ignore_case (bool): Check for uppercase, lowercase and titlecase.
        RETURNS (list): The string keys of other entries with different senses.
        """
        result = []
        key = key if isinstance(key, str) else self.strings[key]
        word, orig_sense = self.split_key(key)
        versions = [word, word.upper(), word.title()] if ignore_case else [word]
        for text in versions:
            for sense in self.senses:
                new_key = self.make_key(text, sense)
                if sense != orig_sense and new_key in self:
                    result.append(new_key)
        return result

    def get_best_sense(
        self, word: str, senses: Sequence[str] = tuple(), ignore_case: bool = True
    ) -> Union[str, None]:
        """Find the best-matching sense for a given word based on the available
        senses and frequency counts. Returns None if no match is found.

        word (unicode): The word to check.
        senses (list): Optional list of senses to limit the search to. If not
            set / empty, all senses in the vectors are used.
        ignore_case (bool): Check for uppercase, lowercase and titlecase.
        RETURNS (unicode): The best-matching key or None.
        """
        sense_options = senses or self.senses
        if not sense_options:
            return None
        versions = [word, word.upper(), word.title()] if ignore_case else [word]
        freqs = []
        for text in versions:
            for sense in sense_options:
                key = self.make_key(text, sense)
                if key in self:
                    freq = self.get_freq(key, -1)
                    freqs.append((freq, key))
        return max(freqs)[1] if freqs else None

    def to_bytes(self, exclude: Sequence[str] = tuple()) -> bytes:
        """Serialize a Sense2Vec object to a bytestring.

        exclude (list): Names of serialization fields to exclude.
        RETURNS (bytes): The serialized Sense2Vec object.
        """
        vectors_bytes = self.vectors.to_bytes()
        freqs = list(self.freqs.items())
        data = {"vectors": vectors_bytes, "cfg": self.cfg, "freqs": freqs}
        if "strings" not in exclude:
            data["strings"] = self.strings.to_bytes()
        if "cache" not in exclude:
            data["cache"] = self.cache
        return srsly.msgpack_dumps(data)

    def from_bytes(self, bytes_data: bytes, exclude: Sequence[str] = tuple()):
        """Load a Sense2Vec object from a bytestring.

        bytes_data (bytes): The data to load.
        exclude (list): Names of serialization fields to exclude.
        RETURNS (Sense2Vec): The loaded object.
        """
        data = srsly.msgpack_loads(bytes_data)
        self.vectors = Vectors().from_bytes(data["vectors"])
        self.freqs = dict(data.get("freqs", []))
        self.cfg.update(data.get("cfg", {}))
        if "strings" not in exclude and "strings" in data:
            self.strings = StringStore().from_bytes(data["strings"])
        if "cache" not in exclude and "cache" in data:
            self.cache = data.get("cache", {})
        return self

    def to_disk(self, path: Union[Path, str], exclude: Sequence[str] = tuple()):
        """Serialize a Sense2Vec object to a directory.

        path (unicode / Path): The path.
        exclude (list): Names of serialization fields to exclude.
        """
        path = Path(path)
        self.vectors.to_disk(path)
        srsly.write_json(path / "cfg", self.cfg)
        srsly.write_json(path / "freqs.json", list(self.freqs.items()))
        if "strings" not in exclude:
            self.strings.to_disk(path / "strings.json")
        if "cache" not in exclude and self.cache:
            srsly.write_msgpack(path / "cache", self.cache)

    def from_disk(self, path: Union[Path, str], exclude: Sequence[str] = tuple()):
        """Load a Sense2Vec object from a directory.

        path (unicode / Path): The path to load from.
        exclude (list): Names of serialization fields to exclude.
        RETURNS (Sense2Vec): The loaded object.
        """
        path = Path(path)
        strings_path = path / "strings.json"
        freqs_path = path / "freqs.json"
        cache_path = path / "cache"
        self.vectors = Vectors().from_disk(path)
        self.cfg.update(srsly.read_json(path / "cfg"))
        if freqs_path.exists():
            self.freqs = dict(srsly.read_json(freqs_path))
        if "strings" not in exclude and strings_path.exists():
            self.strings = StringStore().from_disk(strings_path)
        if "cache" not in exclude and cache_path.exists():
            self.cache = srsly.read_msgpack(cache_path)
        return self
