#!/usr/bin/env python3
import math
from pathlib import Path

import numpy as np
import pytest

from semsis.encoder.sentence_encoder import SentenceEncoder
from semsis.kvstore import KVStore
from semsis.retriever.faiss import RetrieverFaiss

TEXT = [
    "They listen to jazz and he likes jazz piano like Bud Powell.",
    "I really like fruites, especially I love grapes.",
    "I am interested in the k-nearest-neighbor search.",
    "The numpy.squeeze() function is used to remove single-dimensional entries from the shape of an array.",
    "This content is restricted.",
]

QUERYS = [
    "I've implemented some k-nearest-neighbor search algorithms.",
    "I often listen to jazz and I have many CDs which Bud Powell played.",
    "I am interested in the k-nearest-neighbor search.",
]

BATCH_SIZE = 2


@pytest.mark.parametrize("representation", ["cls", "avg", "sbert"])
@pytest.mark.parametrize(
    "model", ["bert-base-uncased", "sentence-transformers/all-MiniLM-L6-v2"]
)
def test_end2end(tmp_path: Path, model, representation):
    # 1. Encode the sentences and store in a key--value store.
    encoder = SentenceEncoder.build(model, representation)
    dim = encoder.get_embed_dim()
    num_sentences = len(TEXT)
    kvstore_path = tmp_path / "kv.bin"
    with KVStore.open(kvstore_path, mode="w") as kvstore:
        # Initialize the kvstore.
        kvstore.new(dim)
        for i in range(math.ceil(num_sentences / BATCH_SIZE)):
            b, e = i * BATCH_SIZE, min((i + 1) * BATCH_SIZE, num_sentences)
            sentence_vectors = encoder.encode(TEXT[b:e]).numpy()
            kvstore.add(sentence_vectors)

    # 2. Read the KVStore and build the kNN index.
    with KVStore.open(kvstore_path, mode="r") as kvstore:
        retriever = RetrieverFaiss.build(RetrieverFaiss.Config(dim))
        retriever.train(kvstore.key[:])
        retriever.add(kvstore.key[:], kvstore.value[:])

    # 3. Save the index.
    index_path = tmp_path / "index.bin"
    cfg_path = tmp_path / "cfg.yaml"
    retriever.save(index_path, cfg_path)
    del retriever

    # 4. Query.
    retriever = RetrieverFaiss.load(index_path, cfg_path)
    query_vectors = encoder.encode(QUERYS).numpy()
    distances, indices = retriever.search(query_vectors, k=1)

    assert indices.squeeze(1).tolist() == [2, 0, 2]
    assert np.isclose(distances[2, 0], 0.0)


if __name__ == "__main__":
    pytest.main()
