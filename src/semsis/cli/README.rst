CLI scripts
###########

Example for text search
=======================

1. Encode the text corpus using `LaBSE <https://huggingface.co/sentence-transformers/LaBSE>`_
   and store key--value pairs by :code:`store_kv.py`.

.. code:: bash

    python store_kv.py \
          --input corpus.txt \
          --output kv.bin \
          --model sentenece-transformers/LaBSE --representation sbert \
          --batch-size 128 \
          --fp16 \
          --workers 16 --chunk-size 10000

2. Next, build a retriever from the key--value store using :code:`build_retriever.py`.

.. code:: bash

    python build_retriever.py \
          --kvstore kv.bin \
          --index-path index.bin \   # The index will be saved in this path.
          --config-path index.yaml \ # The index configuration will be saved in this path.
          --backend faiss-cpu \
          --chunk-size 100000 \      # Number of vectors to be added at a time.
          --train-size 100000 \      # Number of training vectors.
          --ivf-nlists 4096 \        # Number of centroids for an inverted file index (IVF). <=0 does not use IVF.
          --hnsw-nlinks 32 \         # Number of edges in HNSW. <=0 does not use HNSW for the index or the coarse quantizer.
          --pq-nblocks 64 \          # Number of sub-vectors divided by PQ. <=0 does not use PQ.
          --pca --pca-dim 256        # Reduce the dimension by PCA.

You can add entries from multiple key--value stores for two purposes.

One is for the multi-key search.
This registers multiple keys that have the same line number in the text files as a same record.
For example, parallel sentences of neighboring cases can be searched if you encode each source side and target side of a parallel corpus using a multilingual sentence encoder, 

The other one is for a large key--value store.
This regards multiple key--value stores as sharded and each one is appended sequentially by :code:`--append-sequential` option.
When adding an example to the index, the ID value is simply shifted by :code:`+ len(retriever)` and the example is appended from the end of the index.

3. Query by :code:`query_interactive.py`.

In the default, the query text is read from the standard input.

.. code:: bash

    python query_interactive.py \
          --index-path index.bin \
          --config-path index.yaml \
          --model sentenece-transformers/LaBSE --representation sbert \
          --gpu-encode --fp16 \  # Use CUDA for encoding the query text.
          --buffer-size 1 \      # 1 means unbuffered.
          --topk 10

If you input the query text interactively, I recommend to use `rlwrap <https://github.com/hanslub42/rlwrap>`_.

:code:`--input` option reads the query text from a file instead of the standard input.
If you use this option, I recommend to increase :code:`--buffer-size` to speed up.
