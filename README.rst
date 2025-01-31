SEMSIS: Semantic Similarity Search
##################################

|PyPI version| |PyPI license| |CI|

.. |PyPI version| image:: https://img.shields.io/pypi/v/semsis.svg
   :target: https://pypi.python.org/pypi/semsis
.. |PyPI license| image:: https://img.shields.io/pypi/l/semsis.svg
   :target: https://pypi.python.org/pypi/semsis
.. |CI| image:: https://github.com/de9uch1/semsis/actions/workflows/ci.yaml/badge.svg
   :target: https://github.com/de9uch1/semsis

*Semsis* is a library for semantic similarity search.
It is designed to focus on the following goals:

- Simplicity: This library is not rich or complex and implements only the minimum necessary for semantic search.
- Maintainability: Unit tests, docstrings, and type hints are all available.
- Extensibility: Additional code can be implemented as needed easily.
- Efficiency: Billion-scale indexes can be constructed efficiently. See `docs/technical_notes.rst <./docs/technical_notes.rst>`_ for details.


REQUIREMENTS
============

- `faiss <https://github.com/facebookresearch/faiss>`_ (see `INSTALL.md <https://github.com/facebookresearch/faiss/blob/main/INSTALL.md>`_)
- The other requirements are defined in :code:`pyproject.toml` and can be installed via :code:`pip install ./`.

INSTALLATION
============

via pip:

.. code:: bash

    pip install semsis

from the source:

.. code:: bash

    git clone https://github.com/de9uch1/semsis.git
    cd semsis/
    pip install ./

from the source with uv:

.. code:: bash

    git clone https://github.com/de9uch1/semsis.git
    cd semsis/
    uv sync

USAGE
=====

Case 1: Use semsis as API
-------------------------

You can see the example of text search in `end2end_test.py <./tests/end2end_test.py>`_.

Note that this example is not optimized for billion-scale index construction.
If you find the efficient implementation, please see `src/semsis/cli/README.rst <./src/semsis/cli/README.rst>`_.

1. Encode the sentences and store in a key--value store.

.. code:: python

    from semsis.encoder import SentenceEncoder
    from semsis.kvstore import KVStore
    from semsis.retriever import RetrieverFaissCPU
    import math
    import numpy as np

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
        retriever = RetrieverFaissCPU.build(RetrieverFaissCPU.Config(dim))
        retriever.train(kvstore.key[:])
        retriever.add(kvstore.key[:], kvstore.value[:])

    retriever.save(INDEX_PATH, INDEX_CONFIG_PATH)

3. Query.

.. code:: python

    retriever = RetrieverFaissCPU.load(INDEX_PATH, INDEX_CONFIG_PATH)
    query_vectors = encoder.encode(QUERYS).numpy()
    distances, indices = retriever.search(query_vectors, k=1)

    assert indices.squeeze(1).tolist() == [2, 0, 2]
    assert np.isclose(distances[2, 0], 0.0)


Case 2: Use semsis as command line scripts
------------------------------------------

Command line scripts are carefully designed to run efficiently for the billion-scale search.
See `src/semsis/cli/README.rst <./src/semsis/cli/README.rst>`_.


LICENSE
=======
This library is published under the MIT-license.
