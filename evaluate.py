import sys
import numpy as np
import matplotlib.pyplot as plt
from peer import Peer
from dht import BaselineDHT, BloomFilterDHT
import random
import csv

NETWORK_HOP_TIME = 10 
BANDWIDTH_RATE = 1024 * 1024 / 1000
NUM_TRIALS = 5  

def load_queries_from_tsv(filepath, num_queries=None, random_seed=None):
    queries = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if len(row) > 1:
                    queries.append(row[1])
    except FileNotFoundError:
        print(f"Warning: Query file not found at {filepath}. Using synthetic queries.")
        return [], []

    if not queries:
        return [], []

    if random_seed is not None:
        random.seed(random_seed)

    from collections import Counter
    word_counts = Counter()
    for q in queries:
        word_counts.update(q.split())

    real_vocab = [word for word, count in word_counts.most_common()]

    if num_queries is not None and len(queries) > num_queries:
        queries = random.sample(queries, num_queries)
    elif num_queries is not None and len(queries) < num_queries:
        queries = random.choices(queries, k=num_queries)

    return queries, real_vocab

def generate_dataset(num_peers, num_docs_per_peer, vocab_size, real_vocab=None):

    if real_vocab:
        words = list(dict.fromkeys(real_vocab))
        vocab_size = len(words)  
    else:
        words = [f"word_{i}" for i in range(vocab_size)]

    current_vocab_size = len(words)

    total_words_to_generate = num_docs_per_peer * num_peers * 10 
    word_indices = (np.random.zipf(a=1.1, size=total_words_to_generate) - 1) % current_vocab_size
    
    documents = {i: [] for i in range(num_peers)}
    # all_keywords = [] 
    
    word_idx_counter = 0
    for peer_id in range(num_peers):
        for _ in range(num_docs_per_peer):
            doc_words = []
            for _ in range(10): 
                if word_idx_counter < len(word_indices):
                    doc_words.append(words[word_indices[word_idx_counter]])
                    word_idx_counter += 1
                else:
                    doc_words.append(random.choice(words)) 
            
            documents[peer_id].append(" ".join(doc_words))
            # all_keywords.extend(doc_words) 

    # print(f"DEBUG: Total documents generated: {sum(len(d) for d in documents.values())}")
    # print(f"DEBUG: Sample of generated keywords in docs: {random.sample(list(set(w for dlist in documents.values() for d in dlist for w in d.split())), min(10, sum(len(dlist) for dlist in documents.values())))}")
    return documents, [] 

def run_baseline_sim(queries, num_peers=100, num_docs_per_peer=50, vocab_size=1000, real_vocab=None):
    peers = [Peer(i) for i in range(num_peers)]
    dht = BaselineDHT(peers)
    documents, _ = generate_dataset(num_peers, num_docs_per_peer, vocab_size, real_vocab)

    for peer_id, docs in documents.items():
        for doc in docs:
            peers[peer_id].add_document(doc)
        peers[peer_id].create_index()
        for keyword in peers[peer_id].index:
            dht.publish(keyword, peer_id)

    total_bandwidth = 0
    total_latency = 0
    successful_queries = 0

    for query in queries:
        querying_peer_id = random.randint(0, num_peers - 1)
        query_terms = [term for term in query.split() if term]
        if not query_terms:
            continue

        posting_lists = []
        query_bandwidth = 0
        query_latency = 0

        for term in query_terms:
            query_latency += NETWORK_HOP_TIME * 2

            responsible_peer_id, posting_list = dht.lookup(term)

            if posting_list is not None and len(posting_list) > 0:
                posting_lists.append(set(posting_list))
                posting_list_size = max(4, len(posting_list) * 4)
                transfer_time = posting_list_size / BANDWIDTH_RATE

                query_bandwidth += posting_list_size
                query_latency += transfer_time

        if posting_lists:
            intersection = posting_lists[0]
            for pl in posting_lists[1:]:
                intersection = intersection.intersection(pl)

            if len(intersection) > 0:
                successful_queries += 1

            total_bandwidth += query_bandwidth
            total_latency += query_latency

    return total_bandwidth, total_latency, (successful_queries / len(queries)) * 100 if len(queries) > 0 else 0

def run_bloom_filter_sim(queries, num_peers=100, num_docs_per_peer=50, vocab_size=1000, real_vocab=None):
    peers = [Peer(i) for i in range(num_peers)]
    bloom_capacity = num_peers
    dht = BloomFilterDHT(peers, capacity=bloom_capacity, error_rate=0.01)
    documents, _ = generate_dataset(num_peers, num_docs_per_peer, vocab_size, real_vocab)

    for peer_id, docs in documents.items():
        for doc in docs:
            peers[peer_id].add_document(doc)
        peers[peer_id].create_index()
        for keyword in peers[peer_id].index:
            dht.publish(keyword, peer_id)

    total_bandwidth = 0
    total_latency = 0
    successful_queries = 0

    for query in queries:
        querying_peer_id = random.randint(0, num_peers - 1)
        query_terms = [term for term in query.split() if term]
        if not query_terms:
            continue

        bloom_filters = []
        query_bandwidth = 0
        query_latency = 0

        for term in query_terms:
            query_latency += NETWORK_HOP_TIME * 2

            responsible_peer_id, bloom_filter = dht.lookup(term)

            if bloom_filter is not None:
                bloom_filters.append(bloom_filter)
                bloom_filter_size = bloom_filter.num_bits / 8
                transfer_time = bloom_filter_size / BANDWIDTH_RATE

                query_bandwidth += bloom_filter_size
                query_latency += transfer_time

        if bloom_filters:
            candidate_peers = []
            for peer_id in range(num_peers):
                if all(peer_id in bf for bf in bloom_filters):
                    candidate_peers.append(peer_id)

            if len(candidate_peers) > 0:
                successful_queries += 1

            candidate_list_size = len(candidate_peers) * 4
            query_bandwidth += candidate_list_size
            query_latency += candidate_list_size / BANDWIDTH_RATE
            query_latency += NETWORK_HOP_TIME 

            total_bandwidth += query_bandwidth
            total_latency += query_latency

    return total_bandwidth, total_latency, (successful_queries / len(queries)) * 100 if len(queries) > 0 else 0

def run_caching_sim(queries, num_peers=100, num_docs_per_peer=50, vocab_size=1000, real_vocab=None):
    peers = [Peer(i, cache_size=50) for i in range(num_peers)]
    bloom_capacity = num_peers
    dht = BloomFilterDHT(peers, capacity=bloom_capacity, error_rate=0.01)
    documents, _ = generate_dataset(num_peers, num_docs_per_peer, vocab_size, real_vocab)

    for peer_id, docs in documents.items():
        for doc in docs:
            peers[peer_id].add_document(doc)
        peers[peer_id].create_index()
        for keyword in peers[peer_id].index:
            dht.publish(keyword, peer_id)

    total_bandwidth = 0
    total_latency = 0
    cache_hits = 0
    total_term_lookups = 0
    successful_queries = 0
    total_queries = 0

    for query in queries:
        total_queries += 1
        querying_peer_id = random.randint(0, num_peers - 1)
        querying_peer = peers[querying_peer_id]
        query_terms = [term for term in query.split() if term]
        if not query_terms:
            continue

        bloom_filters = []
        query_bandwidth = 0
        query_latency = 0

        for term in query_terms:
            total_term_lookups += 1
            cached_result = querying_peer.get_from_cache(term)

            if cached_result:
                cache_hits += 1
                bloom_filters.append(cached_result)
                query_latency += NETWORK_HOP_TIME  
            else:
                query_latency += NETWORK_HOP_TIME * 2 

                responsible_peer_id, bloom_filter = dht.lookup(term)

                if bloom_filter is not None:
                    bloom_filters.append(bloom_filter)
                    bloom_filter_size = bloom_filter.num_bits / 8
                    transfer_time = bloom_filter_size / BANDWIDTH_RATE

                    query_bandwidth += bloom_filter_size
                    query_latency += transfer_time

                    querying_peer.add_to_cache(term, bloom_filter)

        if bloom_filters:
            candidate_peers = []
            for peer_id in range(num_peers):
                if all(peer_id in bf for bf in bloom_filters):
                    candidate_peers.append(peer_id)

            if len(candidate_peers) > 0:
                successful_queries += 1

            candidate_list_size = len(candidate_peers) * 4
            query_bandwidth += candidate_list_size
            query_latency += candidate_list_size / BANDWIDTH_RATE
            query_latency += NETWORK_HOP_TIME

            total_bandwidth += query_bandwidth
            total_latency += query_latency

    cache_hit_rate = (cache_hits / total_term_lookups) * 100 if total_term_lookups > 0 else 0
    return total_bandwidth, total_latency, cache_hit_rate, (successful_queries / total_queries) * 100 if total_queries > 0 else 0

def run_churn_sim_baseline_or_bloom(queries, num_peers=100, num_docs_per_peer=50, vocab_size=1000, real_vocab=None, churn_rate=10, dht_type=BaselineDHT):
    peers = [Peer(i) for i in range(num_peers)]
    if dht_type == BaselineDHT:
        dht_instance = BaselineDHT(peers)
    elif dht_type == BloomFilterDHT:
        bloom_capacity = num_peers
        dht_instance = BloomFilterDHT(peers, capacity=bloom_capacity, error_rate=0.01)

    documents, _ = generate_dataset(num_peers, num_docs_per_peer, vocab_size, real_vocab)

    for peer_id, docs in documents.items():
        for doc in docs:
            peers[peer_id].add_document(doc)
        peers[peer_id].create_index()
        for keyword in peers[peer_id].index:
            dht_instance.publish(keyword, peer_id)

    total_bandwidth = 0
    total_latency = 0
    successful_queries = 0
    total_queries = 0

    failed_peers_timeline = []

    for i, query in enumerate(queries):
        if i > 0 and i % churn_rate == 0:
            peer_to_fail = random.randint(0, num_peers - 1)
            dht_instance.fail_peer(peer_to_fail)
            failed_peers_timeline.append(peer_to_fail)
            if len(failed_peers_timeline) > 5:
                peer_to_recover = failed_peers_timeline.pop(0)
                dht_instance.recover_peer(peer_to_recover)

        total_queries += 1
        querying_peer_id = random.randint(0, num_peers - 1)
        query_terms = [term for term in query.split() if term]
        if not query_terms:
            continue

        results = []
        query_bandwidth = 0
        query_latency = 0

        for term in query_terms:
            query_latency += NETWORK_HOP_TIME * 2

            responsible_peer_id, result = dht_instance.lookup(term)

            if result is not None:
                results.append(result)

                if dht_type == BaselineDHT:
                    posting_list_size = max(4, len(result) * 4)
                    transfer_time = posting_list_size / BANDWIDTH_RATE
                    query_bandwidth += posting_list_size
                    query_latency += transfer_time

                elif dht_type == BloomFilterDHT:
                    bloom_filter_size = result.num_bits / 8
                    transfer_time = bloom_filter_size / BANDWIDTH_RATE
                    query_bandwidth += bloom_filter_size
                    query_latency += transfer_time

        if results:
            if dht_type == BaselineDHT:
                intersection = set(results[0])
                for result in results[1:]:
                    intersection = intersection.intersection(set(result))

                if len(intersection) > 0:
                    successful_queries += 1

            elif dht_type == BloomFilterDHT:
                candidate_peers = []
                for peer_id in range(num_peers):
                    if all(peer_id in bf for bf in results):
                        candidate_peers.append(peer_id)

                if len(candidate_peers) > 0:
                    successful_queries += 1

                candidate_list_size = len(candidate_peers) * 4
                query_bandwidth += candidate_list_size
                query_latency += candidate_list_size / BANDWIDTH_RATE
                query_latency += NETWORK_HOP_TIME

            total_bandwidth += query_bandwidth
            total_latency += query_latency

    return total_bandwidth, total_latency, (successful_queries / total_queries) * 100 if total_queries > 0 else 0


def run_churn_sim(queries, num_peers=100, num_docs_per_peer=50, vocab_size=1000, real_vocab=None, churn_rate=10):
    peers = [Peer(i, cache_size=50) for i in range(num_peers)]
    bloom_capacity = num_peers
    dht = BloomFilterDHT(peers, capacity=bloom_capacity, error_rate=0.01)
    documents, _ = generate_dataset(num_peers, num_docs_per_peer, vocab_size, real_vocab)

    for peer_id, docs in documents.items():
        for doc in docs:
            peers[peer_id].add_document(doc)
        peers[peer_id].create_index()
        for keyword in peers[peer_id].index:
            dht.publish(keyword, peer_id)

    total_bandwidth = 0
    total_latency = 0
    cache_hits = 0
    total_term_lookups = 0
    successful_queries = 0
    total_queries = 0

    failed_peers_timeline = []

    for i, query in enumerate(queries):
        # induce churn
        if i > 0 and i % churn_rate == 0:
            failed_peer_id = random.randint(0, num_peers - 1)
            dht.fail_peer(failed_peer_id)
            failed_peers_timeline.append(failed_peer_id)
            if len(failed_peers_timeline) > 5:
                peer_to_recover = failed_peers_timeline.pop(0)
                dht.recover_peer(peer_to_recover)

        total_queries += 1
        querying_peer_id = random.randint(0, num_peers - 1)
        querying_peer = peers[querying_peer_id]
        query_terms = [term for term in query.split() if term]
        if not query_terms:
            continue

        bloom_filters = []
        query_bandwidth = 0
        query_latency = 0

        for term in query_terms:
            total_term_lookups += 1
            cached_result = querying_peer.get_from_cache(term)

            if cached_result:
                cache_hits += 1
                bloom_filters.append(cached_result)
                query_latency += NETWORK_HOP_TIME
            else:
                query_latency += NETWORK_HOP_TIME * 2

                responsible_peer_id, bloom_filter = dht.lookup(term)

                if bloom_filter:
                    bloom_filters.append(bloom_filter)
                    bloom_filter_size = bloom_filter.num_bits / 8
                    transfer_time = bloom_filter_size / BANDWIDTH_RATE

                    query_bandwidth += bloom_filter_size
                    query_latency += transfer_time

                    querying_peer.add_to_cache(term, bloom_filter)

        if bloom_filters:
            candidate_peers = []
            for peer_id in range(num_peers):
                if all(peer_id in bf for bf in bloom_filters):
                    candidate_peers.append(peer_id)

            if len(candidate_peers) > 0:
                successful_queries += 1

            candidate_list_size = len(candidate_peers) * 4
            query_bandwidth += candidate_list_size
            query_latency += candidate_list_size / BANDWIDTH_RATE
            query_latency += NETWORK_HOP_TIME

            total_bandwidth += query_bandwidth
            total_latency += query_latency

    cache_hit_rate = (cache_hits / total_term_lookups) * 100 if total_term_lookups > 0 else 0
    query_success_rate = (successful_queries / total_queries) * 100 if total_queries > 0 else 0

    return total_bandwidth, total_latency, cache_hit_rate, query_success_rate

if __name__ == "__main__":

    print(f"Running simulations with {NUM_TRIALS} trials for averaging...")



    queries, real_vocab = load_queries_from_tsv("../data/queries.train.tsv", num_queries=1000, random_seed=42)



    if not queries:

        print("Could not load queries. Exiting.")

        sys.exit(1)

    results = {
        'baseline_fc': [], 'bloom_fc': [], 'caching_fc': [],
        'baseline_lt': [], 'bloom_lt': [], 'caching_lt': [],
        'baseline_mc': [], 'bloom_mc': [], 'caching_mc': [],
        'baseline_hc': [], 'bloom_hc': [], 'caching_hc': []
    }



    for trial in range(NUM_TRIALS):
        print(f"\n{'='*60}")
        print(f"TRIAL {trial + 1}/{NUM_TRIALS}")
        print(f"{'='*60}")

        random.seed(42 + trial)
        np.random.seed(42 + trial)

        print("\n--- Scenario A: Flash Crowd (High Skew) ---")
        skewed_queries = queries[:10] * 100

        bw, lat, succ = run_baseline_sim(skewed_queries, real_vocab=real_vocab)
        results['baseline_fc'].append({'bandwidth': bw, 'latency': lat, 'success': succ})
        print(f"Baseline -> Bandwidth: {bw} bytes, Latency: {lat:.2f} ms")

        bw, lat, succ = run_bloom_filter_sim(skewed_queries, real_vocab=real_vocab)
        results['bloom_fc'].append({'bandwidth': bw, 'latency': lat, 'success': succ})
        print(f"Bloom Filter -> Bandwidth: {bw} bytes, Latency: {lat:.2f} ms")

        bw, lat, cache_hit, succ = run_caching_sim(skewed_queries, real_vocab=real_vocab)
        results['caching_fc'].append({'bandwidth': bw, 'latency': lat, 'cache_hit': cache_hit, 'success': succ})
        print(f"Caching -> Bandwidth: {bw} bytes, Latency: {lat:.2f} ms, Cache Hit: {cache_hit:.2f}%")

        print("\n--- Scenario B: Long Tail (Uniform) ---")
        uniform_queries = random.sample(queries, 1000)

        bw, lat, succ = run_baseline_sim(uniform_queries, real_vocab=real_vocab)
        results['baseline_lt'].append({'bandwidth': bw, 'latency': lat, 'success': succ})
        print(f"Baseline -> Bandwidth: {bw} bytes, Latency: {lat:.2f} ms")

        bw, lat, succ = run_bloom_filter_sim(uniform_queries, real_vocab=real_vocab)
        results['bloom_lt'].append({'bandwidth': bw, 'latency': lat, 'success': succ})
        print(f"Bloom Filter -> Bandwidth: {bw} bytes, Latency: {lat:.2f} ms")

        bw, lat, cache_hit, succ = run_caching_sim(uniform_queries, real_vocab=real_vocab)
        results['caching_lt'].append({'bandwidth': bw, 'latency': lat, 'cache_hit': cache_hit, 'success': succ})
        print(f"Caching -> Bandwidth: {bw} bytes, Latency: {lat:.2f} ms, Cache Hit: {cache_hit:.2f}%")

        print("\n--- Scenario C: Moderate Churn ---")
        bw, lat, succ = run_churn_sim_baseline_or_bloom(skewed_queries, num_peers=100, num_docs_per_peer=50, vocab_size=1000, real_vocab=real_vocab, churn_rate=100, dht_type=BaselineDHT)
        results['baseline_mc'].append({'bandwidth': bw, 'latency': lat, 'success': succ})
        print(f"Baseline -> Bandwidth: {bw} bytes, Latency: {lat:.2f} ms")

        bw, lat, succ = run_churn_sim_baseline_or_bloom(skewed_queries, num_peers=100, num_docs_per_peer=50, vocab_size=1000, real_vocab=real_vocab, churn_rate=100, dht_type=BloomFilterDHT)
        results['bloom_mc'].append({'bandwidth': bw, 'latency': lat, 'success': succ})
        print(f"Bloom Filter -> Bandwidth: {bw} bytes, Latency: {lat:.2f} ms")

        bw, lat, cache_hit, succ = run_churn_sim(skewed_queries, real_vocab=real_vocab, churn_rate=100)
        results['caching_mc'].append({'bandwidth': bw, 'latency': lat, 'cache_hit': cache_hit, 'success': succ})
        print(f"Caching -> Bandwidth: {bw} bytes, Latency: {lat:.2f} ms, Cache Hit: {cache_hit:.2f}%")

        print("\n--- Scenario C: High Churn ---")
        bw, lat, succ = run_churn_sim_baseline_or_bloom(skewed_queries, num_peers=100, num_docs_per_peer=50, vocab_size=1000, real_vocab=real_vocab, churn_rate=10, dht_type=BaselineDHT)
        results['baseline_hc'].append({'bandwidth': bw, 'latency': lat, 'success': succ})
        print(f"Baseline -> Bandwidth: {bw} bytes, Latency: {lat:.2f} ms")

        bw, lat, succ = run_churn_sim_baseline_or_bloom(skewed_queries, num_peers=100, num_docs_per_peer=50, vocab_size=1000, real_vocab=real_vocab, churn_rate=10, dht_type=BloomFilterDHT)
        results['bloom_hc'].append({'bandwidth': bw, 'latency': lat, 'success': succ})
        print(f"Bloom Filter -> Bandwidth: {bw} bytes, Latency: {lat:.2f} ms")

        bw, lat, cache_hit, succ = run_churn_sim(skewed_queries, real_vocab=real_vocab, churn_rate=10)
        results['caching_hc'].append({'bandwidth': bw, 'latency': lat, 'cache_hit': cache_hit, 'success': succ})
        print(f"Caching -> Bandwidth: {bw} bytes, Latency: {lat:.2f} ms, Cache Hit: {cache_hit:.2f}%")

    print(f"\n{'='*60}")
    print("AVERAGED RESULTS ACROSS ALL TRIALS")
    print(f"{'='*60}")

    def calc_stats(data_list, metric):
        values = [d[metric] for d in data_list]
        return np.mean(values), np.std(values)

    print("\n--- Scenario A: Flash Crowd (High Skew) ---")
    baseline_bw_fc, baseline_bw_fc_std = calc_stats(results['baseline_fc'], 'bandwidth')
    baseline_lat_fc, baseline_lat_fc_std = calc_stats(results['baseline_fc'], 'latency')
    print(f"Baseline -> Bandwidth: {baseline_bw_fc:.0f} ± {baseline_bw_fc_std:.0f} bytes, Latency: {baseline_lat_fc:.2f} ± {baseline_lat_fc_std:.2f} ms")

    bloom_bw_fc, bloom_bw_fc_std = calc_stats(results['bloom_fc'], 'bandwidth')
    bloom_lat_fc, bloom_lat_fc_std = calc_stats(results['bloom_fc'], 'latency')
    print(f"Bloom Filter -> Bandwidth: {bloom_bw_fc:.0f} ± {bloom_bw_fc_std:.0f} bytes, Latency: {bloom_lat_fc:.2f} ± {bloom_lat_fc_std:.2f} ms")

    caching_bw_fc, caching_bw_fc_std = calc_stats(results['caching_fc'], 'bandwidth')
    caching_lat_fc, caching_lat_fc_std = calc_stats(results['caching_fc'], 'latency')
    caching_ch_fc, _ = calc_stats(results['caching_fc'], 'cache_hit')
    print(f"Caching -> Bandwidth: {caching_bw_fc:.0f} ± {caching_bw_fc_std:.0f} bytes, Latency: {caching_lat_fc:.2f} ± {caching_lat_fc_std:.2f} ms, Cache Hit: {caching_ch_fc:.2f}%")

    print("\n--- Scenario B: Long Tail (Uniform) ---")
    baseline_bw_lt, baseline_bw_lt_std = calc_stats(results['baseline_lt'], 'bandwidth')
    baseline_lat_lt, baseline_lat_lt_std = calc_stats(results['baseline_lt'], 'latency')
    print(f"Baseline -> Bandwidth: {baseline_bw_lt:.0f} ± {baseline_bw_lt_std:.0f} bytes, Latency: {baseline_lat_lt:.2f} ± {baseline_lat_lt_std:.2f} ms")

    bloom_bw_lt, bloom_bw_lt_std = calc_stats(results['bloom_lt'], 'bandwidth')
    bloom_lat_lt, bloom_lat_lt_std = calc_stats(results['bloom_lt'], 'latency')
    print(f"Bloom Filter -> Bandwidth: {bloom_bw_lt:.0f} ± {bloom_bw_lt_std:.0f} bytes, Latency: {bloom_lat_lt:.2f} ± {bloom_lat_lt_std:.2f} ms")

    caching_bw_lt, caching_bw_lt_std = calc_stats(results['caching_lt'], 'bandwidth')
    caching_lat_lt, caching_lat_lt_std = calc_stats(results['caching_lt'], 'latency')
    caching_ch_lt, _ = calc_stats(results['caching_lt'], 'cache_hit')
    print(f"Caching -> Bandwidth: {caching_bw_lt:.0f} ± {caching_bw_lt_std:.0f} bytes, Latency: {caching_lat_lt:.2f} ± {caching_lat_lt_std:.2f} ms, Cache Hit: {caching_ch_lt:.2f}%")

    print("\n--- Scenario C: Moderate Churn ---")
    baseline_bw_mc, baseline_bw_mc_std = calc_stats(results['baseline_mc'], 'bandwidth')
    baseline_lat_mc, baseline_lat_mc_std = calc_stats(results['baseline_mc'], 'latency')
    print(f"Baseline -> Bandwidth: {baseline_bw_mc:.0f} ± {baseline_bw_mc_std:.0f} bytes, Latency: {baseline_lat_mc:.2f} ± {baseline_lat_mc_std:.2f} ms")

    bloom_bw_mc, bloom_bw_mc_std = calc_stats(results['bloom_mc'], 'bandwidth')
    bloom_lat_mc, bloom_lat_mc_std = calc_stats(results['bloom_mc'], 'latency')
    print(f"Bloom Filter -> Bandwidth: {bloom_bw_mc:.0f} ± {bloom_bw_mc_std:.0f} bytes, Latency: {bloom_lat_mc:.2f} ± {bloom_lat_mc_std:.2f} ms")

    caching_bw_mc, caching_bw_mc_std = calc_stats(results['caching_mc'], 'bandwidth')
    caching_lat_mc, caching_lat_mc_std = calc_stats(results['caching_mc'], 'latency')
    caching_ch_mc, _ = calc_stats(results['caching_mc'], 'cache_hit')
    print(f"Caching -> Bandwidth: {caching_bw_mc:.0f} ± {caching_bw_mc_std:.0f} bytes, Latency: {caching_lat_mc:.2f} ± {caching_lat_mc_std:.2f} ms, Cache Hit: {caching_ch_mc:.2f}%")

    print("\n--- Scenario C: High Churn ---")
    baseline_bw_hc, baseline_bw_hc_std = calc_stats(results['baseline_hc'], 'bandwidth')
    baseline_lat_hc, baseline_lat_hc_std = calc_stats(results['baseline_hc'], 'latency')
    print(f"Baseline -> Bandwidth: {baseline_bw_hc:.0f} ± {baseline_bw_hc_std:.0f} bytes, Latency: {baseline_lat_hc:.2f} ± {baseline_lat_hc_std:.2f} ms")

    bloom_bw_hc, bloom_bw_hc_std = calc_stats(results['bloom_hc'], 'bandwidth')
    bloom_lat_hc, bloom_lat_hc_std = calc_stats(results['bloom_hc'], 'latency')
    print(f"Bloom Filter -> Bandwidth: {bloom_bw_hc:.0f} ± {bloom_bw_hc_std:.0f} bytes, Latency: {bloom_lat_hc:.2f} ± {bloom_lat_hc_std:.2f} ms")

    caching_bw_hc, caching_bw_hc_std = calc_stats(results['caching_hc'], 'bandwidth')
    caching_lat_hc, caching_lat_hc_std = calc_stats(results['caching_hc'], 'latency')
    caching_ch_hc, _ = calc_stats(results['caching_hc'], 'cache_hit')
    print(f"Caching -> Bandwidth: {caching_bw_hc:.0f} ± {caching_bw_hc_std:.0f} bytes, Latency: {caching_lat_hc:.2f} ± {caching_lat_hc_std:.2f} ms, Cache Hit: {caching_ch_hc:.2f}%")

    print(f"\n{'='*60}")
    print("GENERATING PLOTS")
    print(f"{'='*60}")

    labels = ['Flash Crowd', 'Long Tail', 'Moderate Churn', 'High Churn']

    baseline_bandwidths_plot = [baseline_bw_fc, baseline_bw_lt, baseline_bw_mc, baseline_bw_hc]
    bloom_bandwidths_plot = [bloom_bw_fc, bloom_bw_lt, bloom_bw_mc, bloom_bw_hc]
    caching_bandwidths_plot = [caching_bw_fc, caching_bw_lt, caching_bw_mc, caching_bw_hc]

    baseline_bw_errors = [baseline_bw_fc_std, baseline_bw_lt_std, baseline_bw_mc_std, baseline_bw_hc_std]
    bloom_bw_errors = [bloom_bw_fc_std, bloom_bw_lt_std, bloom_bw_mc_std, bloom_bw_hc_std]
    caching_bw_errors = [caching_bw_fc_std, caching_bw_lt_std, caching_bw_mc_std, caching_bw_hc_std]

    baseline_latencies_plot = [baseline_lat_fc, baseline_lat_lt, baseline_lat_mc, baseline_lat_hc]
    bloom_latencies_plot = [bloom_lat_fc, bloom_lat_lt, bloom_lat_mc, bloom_lat_hc]
    caching_latencies_plot = [caching_lat_fc, caching_lat_lt, caching_lat_mc, caching_lat_hc]

    baseline_lat_errors = [baseline_lat_fc_std, baseline_lat_lt_std, baseline_lat_mc_std, baseline_lat_hc_std]
    bloom_lat_errors = [bloom_lat_fc_std, bloom_lat_lt_std, bloom_lat_mc_std, bloom_lat_hc_std]
    caching_lat_errors = [caching_lat_fc_std, caching_lat_lt_std, caching_lat_mc_std, caching_lat_hc_std]

    x = np.arange(len(labels))
    width = 0.25

    # Bandwidth Plot 
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    ax1.bar(x - width, baseline_bandwidths_plot, width, label='Baseline')
    ax1.bar(x, bloom_bandwidths_plot, width, label='Bloom Filter')
    ax1.bar(x + width, caching_bandwidths_plot, width, label='Caching')
    ax1.set_ylabel('Total Bandwidth (bytes)', fontsize=12)
    ax1.set_title(f'Bandwidth Consumption Comparison (Averaged over {NUM_TRIALS} trials)', fontsize=14, fontweight='bold')
    ax1.set_yscale('log')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=11)
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('../img/bandwidth_comparison.png', dpi=300)
    print("✓ Bandwidth plot saved to ../img/bandwidth_comparison.png")

    fig2, ax2 = plt.subplots(figsize=(12, 7))
    ax2.bar(x - width, baseline_latencies_plot, width, label='Baseline')
    ax2.bar(x, bloom_latencies_plot, width, label='Bloom Filter')
    ax2.bar(x + width, caching_latencies_plot, width, label='Caching')
    ax2.set_ylabel('Total Latency (ms)', fontsize=12)
    ax2.set_title(f'Query Latency Comparison (Averaged over {NUM_TRIALS} trials)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=11)
    ax2.legend(fontsize=11)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('../img/latency_comparison.png', dpi=300)
    print("✓ Latency plot saved to ../img/latency_comparison.png")

    print(f"\n{'='*60}")
    print("ALL SIMULATIONS COMPLETE")
    print(f"{'='*60}")
    print(f"Ran {NUM_TRIALS} trials per scenario")
    print("Plots saved to final-project/img/ directory with averaged results and error bars")
