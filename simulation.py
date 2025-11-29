"""
This module contains the main simulation logic.
"""
import sys
import numpy as np
from peer import Peer
from dht import DHT

class Simulation:
    def __init__(self, num_peers):
        self.num_peers = num_peers
        self.peers = []
        self.dht = None

    def setup(self):
        """
        Set up the simulation environment.
        """
        # Create peers
        for i in range(self.num_peers):
            self.peers.append(Peer(peer_id=i))

        # Create DHT
        self.dht = DHT(self.peers)

        # Assign documents to peers
        documents = {
            0: ["apple banana orange", "apple grape"],
            1: ["orange grape kiwi", "kiwi strawberry"],
            2: ["apple strawberry blueberry", "blueberry raspberry"],
            3: ["grape kiwi strawberry", "blueberry raspberry apple"],
        }

        for peer_id, docs in documents.items():
            if peer_id < self.num_peers:
                for doc in docs:
                    self.peers[peer_id].add_document(doc)
                self.peers[peer_id].create_index()

        # Publish keywords to DHT
        for peer in self.peers:
            for keyword in peer.index:
                self.dht.publish(keyword, peer.peer_id)


    def run(self, num_queries=100):
        """
        Run the simulation.
        """
        # Create a skewed query distribution
        keywords = ["apple", "banana", "orange", "grape", "kiwi", "strawberry", "blueberry", "raspberry"]
        # Skewed distribution, 'apple' will be most popular
        keyword_distribution = np.random.zipf(a=2, size=num_queries) % len(keywords)

        queries = [keywords[i] for i in keyword_distribution]

        cache_hits = 0
        total_queries = 0

        for keyword_to_query in queries:
            total_queries += 1
            querying_peer_id = np.random.randint(0, self.num_peers)
            querying_peer = self.peers[querying_peer_id]

            # Check cache first
            cached_result = querying_peer.get_from_cache(keyword_to_query)
            if cached_result:
                cache_hits += 1
                # print(f"Query for '{keyword_to_query}' from peer {querying_peer_id}: Cache hit!")
            else:
                # print(f"Query for '{keyword_to_query}' from peer {querying_peer_id}: Cache miss.")
                responsible_peer_id, bloom_filter = self.dht.lookup(keyword_to_query)
                if bloom_filter:
                    querying_peer.add_to_cache(keyword_to_query, bloom_filter)

        cache_hit_rate = (cache_hits / total_queries) * 100
        print(f"Cache hit rate: {cache_hit_rate:.2f}%")


if __name__ == "__main__":
    sim = Simulation(num_peers=10)
    sim.setup()
    sim.run()
