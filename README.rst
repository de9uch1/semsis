SEMSIS: Semantic Similarity Search
##################################

*Semsis* is a library for semantic similarity search.
It is designed to focus on the following goals:

- Simplicity: This library is not rich or complex and implements only the minimum necessary for semantic search.
- Quality: Unit tests, docstrings, and type hints are all available.
- Extensibility: Additional code can be implemented as needed, e.g., to use arbitrary search engines.
- Efficiency: e.g., our :code:`RetrieverFaissGPU` overrides some methods of faiss for efficient computation.


REQUIREMENTS
============

- `faiss <https://github.com/facebookresearch/faiss>`_ (see `INSTALL.md <https://github.com/facebookresearch/faiss/blob/main/INSTALL.md>`_)
- The other requirements are defined in :code:`pyproject.toml` and can be installed via :code:`pip install ./`.

INSTALLATION
============

.. code:: bash

    git clone https://github.com/de9uch1/semsis.git
    cd semsis/
    pip install ./

USAGE
=====

You can see the example of text search in `end2end_test.py <./tests/end2end.py>`_.

1. Encode the sentences and store in a key--value store.

.. code:: python

    from semsis.encoder import SentenceEncoder
    from semsis.kvstore import KVStore
    from semsis.retriever import RetrieverFaiss

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
    KVSTORE_PATH = "./kv.bin"
    INDEX_PATH = "./index.bin"
    INDEX_CONFIG_PATH = "./cfg.yaml"

    MODEL = "bert-base-uncased"
    REPRESENTATION = "avg"
    BATCH_SIZE = 2

    encoder = SentenceEncoder.build(MODEL, REPRESENTATION)
    dim = encoder.get_embed_dim()
    num_sentences = len(TEXT)
    with KVStore.open(KVSTORE_PATH, mode="w") as kvstore:
        # Initialize the kvstore.
        kvstore.new(dim)
        for i in range(math.ceil(num_sentences / BATCH_SIZE)):
            b, e = i * BATCH_SIZE, min((i + 1) * BATCH_SIZE, num_sentences)
            sentence_vectors = encoder.encode(TEXT[b:e]).numpy()
            kvstore.add(sentence_vectors)

2. Next, read the key--value store and build the kNN index.

.. code:: python

    with KVStore.open(KVSTORE_PATH, mode="r") as kvstore:
        retriever = RetrieverFaiss.build(RetrieverFaiss.Config(dim))
        retriever.train(kvstore.key[:])
        retriever.add(kvstore.key[:], kvstore.value[:])

    retriever.save(INDEX_PATH, INDEX_CONFIG_PATH)

3. Query.

.. code:: python

    retriever = RetrieverFaiss.load(index_path, cfg_path)
    query_vectors = encoder.encode(QUERYS).numpy()
    distances, indices = retriever.search(query_vectors, k=1)

    assert indices.squeeze(1).tolist() == [2, 0, 2]
    assert np.isclose(distances[2, 0], 0.0)
