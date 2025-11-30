# p2p dht simulation

quick sim of a peer-to-peer distributed hash table with bloom filters and caching for a class project.

note that 

## why

comparing different indexing strategies (baseline DHT, bloom filter DHT, and caching) across different query patterns and network conditions.

## run it

```bash
pip install pybloom-live numpy matplotlib
python evaluate.py
```

needs the MSMARCO query dataset in `../data/queries.train.tsv`

## dataset

uses queries from [MS MARCO](https://microsoft.github.io/msmarco/) - specifically the training queries

download: https://msmarco.blob.core.windows.net/msmarcoranking/queries.train.tsv

put it in a `data/` folder one level up from src
