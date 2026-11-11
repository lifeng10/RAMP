# RAMP

Implementation of RAMP for privacy-preserving range-filtered approximate nearest neighbor search, including dimension selection, encrypted range filtering, HNSW/IVF search, candidate reuse, and client reranking.

Tested on Ubuntu 26.04.

## Required Libraries

- [hnswlib](https://github.com/nmslib/hnswlib) 0.8.0
- [RaBitQ-Library](https://github.com/VectorDB-NTU/RaBitQ-Library) 0.3.6, including the compiled `rabitq_core` library
- [Faiss](https://github.com/facebookresearch/faiss) (CPU C++ library)
- [NumPy](https://numpy.org/) >= 1.24
- [h5py](https://www.h5py.org/) >= 3.8 (COCO conversion)

## Build & Compile

Run from the RAMP directory, replacing `/path/to/...` with your library paths:

```bash
cmake -S . -B ../ramp-build -DCMAKE_BUILD_TYPE=Release \
    -DHNSWLIB_ROOT=/path/to/hnswlib \
    -DRABITQ_ROOT=/path/to/RaBitQ-Library \
    -DFAISS_ROOT=/path/to/faiss \
    -DBLA_VENDOR=OpenBLAS
cmake --build ../ramp-build -j2
```

This produces the executable `../ramp-build/ramp`.

## Working Directory

Run all commands from `RAMP/`. Paths beginning with `../` refer to its parent
directory, so `data/` and `RAMP/` are at the same level:

```text
working-directory/
├── RAMP/                 # Run commands here
├── data/
│   ├── sift/             # Extracted SIFT files
│   │   ├── sift_base.fvecs
│   │   ├── sift_query.fvecs
│   │   └── sift_learn.fvecs
│   └── sift100k/         # Converted data
├── ramp-build/           # Compiled executable
├── ramp-input/           # Prepared RAMP inputs
└── ramp-results/         # Query results
```

Create `data/sift/` and place the extracted SIFT files there. The other data and
output directories are created by the commands below and the build commands above.

## Usage

Use `convert_data.py` to convert SIFT, SIFT1M, COCO, or FANNS100K into RAMP inputs.
We use the [SIFT dataset](https://huggingface.co/datasets/zjd2000/ANN_SIFT/resolve/main/sift.tar.gz?download=true)
as an example. Place the extracted `.fvecs` files in `../data/sift/` and preprocess
the data:

```bash
# Convert SIFT data and generate attributes and range queries.
python3 convert_data.py sift ../data/sift --output ../data/sift100k

# Select the search dimension and prepare RAMP inputs.
python3 prepare.py \
    --database ../data/sift100k/database.npy --queries ../data/sift100k/queries.npy \
    --attributes ../data/sift100k/attributes.npy --requests ../data/sift100k/requests.csv \
    --public-vectors ../data/sift100k/public.npy \
    --public-queries ../data/sift100k/public_queries.npy \
    --pca-vectors ../data/sift100k/pca.npy \
    --scale 1024 --gamma 1 --output ../ramp-input
```

This prepares 100K database vectors, 500 queries, public data, and range queries
matching 10% of the database. Run RAMP:

```bash
OPENBLAS_NUM_THREADS=1 ../ramp-build/ramp \
    --input ../ramp-input --output ../ramp-results \
    --scan-threshold 0 --hnsw-threshold 86602 --repeats 3
```

Timing records (in milliseconds) and ranked result IDs are saved in
`../ramp-results/queries.csv` and `../ramp-results/results.csv`, respectively.
Use new output directories when rerunning.

## Datasets used in the paper

| Dataset | Vectors | Dimensions | Download |
| --- | ---: | ---: | --- |
| SIFT | 100K | 128 | 100K subset of [SIFT1M](http://corpus-texmex.irisa.fr/) |
| SIFT1M | 1M | 128 | [TexMex archive](ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz) |
| COCO | 100K | 512 | [COCO image vectors](https://github.com/fabiocarrara/str-encoders/releases/download/v0.1.3/coco-i2i-512-angular.hdf5) |
| FANNS100K | 100K | 4096 | [arxiv-for-fanns-medium](https://huggingface.co/datasets/SPCL/arxiv-for-fanns-medium) |

For the other datasets, use the following conversion commands:

```bash
python3 convert_data.py sift1m ../data/sift --output ../data/sift1m
python3 convert_data.py coco ../data/coco-i2i-512-angular.hdf5 --output ../data/coco
python3 convert_data.py fanns100k ../data/fanns-medium \
    --public-source ../data/fanns-large/database_vectors.fvecs --output ../data/fanns100k
```

FANNS100K reads `database_vectors.fvecs`, `query_vectors.fvecs`, and
`database_attributes.jsonl` from [arxiv-for-fanns-medium](https://huggingface.co/datasets/SPCL/arxiv-for-fanns-medium).
Public vectors come from [arxiv-for-fanns-large](https://huggingface.co/datasets/SPCL/arxiv-for-fanns-large),
excluding evaluation vectors. Its `update_date` values are preserved.
COCO and FANNS100K vectors are normalized during conversion.

Reuse the preparation and run commands with the appropriate dataset paths,
scale, and calibrated thresholds.
Set `--scale 1024` for SIFT/SIFT1M and `--scale 1.001` for COCO/FANNS100K.
The converter also creates 200 separate calibration queries. For threshold
selection, set several range sizes with `--matching`, then use `ramp --calibrate`
and `select_thresholds.py`. Conversion options and the calibration steps are
described by `python3 convert_data.py --help`.
