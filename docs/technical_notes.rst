Goals
#####

This library especially focuses on the following goals:

- Create key vectors yourself without benchmark precomputed vector datasets.
- Scale in speed and memory / storage capacity from small to large datasets.
- Run efficiently from vector computation to index construction on GPU machines.

I mainly assume an environment in which vector computation and index construction are performed on a GPU machine and searching is performed on a CPU machine.
This library can perform index construction processes efficiently.

On the other hand, existing k-nearest-neighbor search libraries, such as faiss, are quite efficient during search; thus, semsis uses them without additional implementations in searching.

Details of each component
#########################

This section describes the technical details of each component, assuming a similar sentence search task.

Vector computation
==================

To begin with, we need to compute the vector representation.
Unlike in training, data can be fed into a model in any order.
Therefore, semsis minimizes the number of padding tokens, which increases the number of tokens that can be computed in the same amount of time.
This can be implemented by tokenizing each sentence, sorting by their lengths, and then packing them into a mini-batch from longer sentence to shorter sentence.
In semsis, the sentence lengths are calculated using :code:`concurrent.futures` for multi-processing.

Then, we store the vector representations.
In this time, both I/O speed and the size of the stored file can be issues.
For example, a 1024-dimensional vector of 32-bit float has a size of 4 KiB. 1 billion vectors consumes 3.8 TiB.
For faster write, semsis employs Hierarchical Data Format 5 (HDF5) which stores large vectors efficiently.
And the large size problem can be solved by dividing the data and indexing each shard.
For instance, if you can divide a dataset into two shards, semsis can construct an index with half the data firstly, delete the first half vector file,
and then compute the second half vectors and adding them to the index.

The overall computation speed can also be improved by dividing the vector computation into multiple shards and computing vectors on multiple GPUs.

Index construction
==================

This section varies depending on the index architecture,
but this document assumes the :code:`OPQ+IVFPQ_HNSW` index,
which have been achieved the state-of-the-art performance in billion-scale kNN search.

Training
--------

IVFPQ training is decomposed into IVF training, i.e., k-means clustering,
and PQ training from the residual vectors between a data vector and its nearest neighbor centroid vector.
In IVF training, semsis assigns cluster IDs using multiple GPUs.
This speeds up the iterative cluster ID assignment in the k-means algorithm.
The reason why use multi GPUs is that IVF learns the L centroids from D-dimensional vectors.
When assigning cluster IDs, IVF computes distance matrix between `L x D` and `N x D`, which would be large, so the assignment index is sharded.

Then, PQ is trained from the residual vectors on a single GPU.
PQ splits input vectors into Dsub-dimensional sub-vectors and assigns quantization codes in each sub-space.
PQ also performs k-means clustering but faiss trains PQ from sampled vectors and does not uses all vectors.
In addition, the memory footprint is smaller (`Ksub x Dsub`, where Ksub is the codebook size and typically =256).
Therefore, semsis trains PQ on a single GPU.

OPQ rotates the input vectors as a pre-transform and the rotation matrix is trained to minimize the PQ reconstruction error.
The rotation matrix is obtained by solving Procrustes problem such as :math:`min_R || R\bm{x} - \mathrm{Decode}(\mathrm{Encode}(R\bm{x})) ||` s.t. :math:`R^\top R = I`.
Here, since there is no PQ codebook to Encode() and Decode(), the objective is obtained by calculating the following alternately:

- Fix R and learn PQ
- Fix PQ and learn R

Thus, in OPQ training, PQ codebook is trained multiple times.
For this, semsis runs PQ training of each OPQ training iteration on a GPU.

Addition
--------

Before adding vectors to the index, OPQ rotates the input vectors by multiplying a square matrix.
Semsis accelerates this matrix multiplying using CUDA.

Next, the rotated vectors are added to an IVFPQ index.
IVFPQ_HNSW finds the nearest neighbor centroid by beam search instead of calculating the distances to all centroids for faster search.
However, it is run on CPUs, because it greedily traverses the graph to search for the shorter paths and hard to implement on CUDA.
While the search is fast enough, the construction is slower because large batched data is processed on the CPU,
so semsis searches the nearest neighbor cluster on the GPU when adding vectors.

In addition, to maximize efficiency, the data is added by sharding the IVFPQ index on multiple GPUs.
However, since it is difficult to put all the large data on the GPU memory,
semsis interatively add vectors by the following procedure:

- Add vectors to the sharded multi-GPU IVFPQ
- Transfer them on the CPU
- Copy and merge them to the master index
- Delete the data on the GPU IVFPQ
