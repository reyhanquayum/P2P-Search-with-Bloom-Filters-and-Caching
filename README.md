# Distributed P2P Search Engine Simulation

I built this simulation project as the final project for CS 6675: Advanced Internet Systems and Applications at Georgia Tech. I have always been interested in decentralized alternatives to centralized search giants, but these systems often fail due to the "feasibility gap" which is the massive bandwidth cost required to query distributed indices. This project explores whether we can bridge that gap by optimizing the architecture with probabilistic data structures and edge caching.

The simulation was designed to quantify the "feasibility gap" in P2P search, specifically addressing the trade-offs between bandwidth efficiency, latency, and resilience under network skew. It compares a naive DHT baseline against two architectural refinements: **Bloom Filters** (for compressed set intersection) and **Distributed LRU Caching**.

## Architecture & Implementation

*   **Chord DHT:** Uses consistent hashing (SHA-1) to map keywords to node IDs, creating a structured overlay network for key-value lookups.
*   **Bloom Filters (Refinement 1):** Replaces raw posting list transfers with Bloom Filters (via `pybloom-live`). This allows peers to compute set intersections locally using probabilistic data structures, significantly reducing the bytes transferred over the wire.
*   **Distributed Caching (Refinement 2):** Implements an LRU (Least Recently Used) cache policy at the peer level using `OrderedDict`. This layer is designed to absorb traffic from "Flash Crowd" events where query distributions follow a Zipfian power law.
*   **Churn Simulation:** The system models dynamic network conditions where peers randomly fail and rejoin, testing the robustness of the index and the staleness of the cache.

## Results

Based on simulations running 1,000 queries on a 100-node network using the MS MARCO dataset:

1.  **Caching Performance:** In high-skew "Flash Crowd" scenarios, the caching layer reduced bandwidth consumption by **58%** and latency by **8%** compared to the baseline.
2.  **Bloom Filter Trade-off:** While Bloom Filters successfully reduced bandwidth by **33-41%**, they introduced a **~9% latency penalty**. This is due to the multi-roundtrip protocol required to fetch filters for every query term before computing the intersection.
3.  **Churn Resilience:** Under high network churn (peers failing every 10 queries), Bloom filters degraded due to "ghost peer" false positives. Caching proved more robust, maintaining a **34% hit rate** even with unstable nodes.

## Setup

### Dependencies

Requires Python 3.10+.

```bash
pip install pybloom-live numpy matplotlib
```

### Dataset

The simulation uses real query patterns from the **MS MARCO** dataset.

1.  Download the training queries: [queries.train.tsv](https://msmarco.blob.core.windows.net/msmarcoranking/queries.train.tsv)
2.  Place the file in a `data/` directory one level up from the source code (e.g., `../data/queries.train.tsv`)

### Execution

The evaluation script runs three scenarios: **Flash Crowd** (High Skew), **Long Tail** (Uniform), and **Network Churn**.

```bash
python evaluate.py
```

Upon completion, the script calculates the Bandwidth Efficiency Ratio and 95th Percentile Latency, and saves visualization plots to the `../img/` directory.
