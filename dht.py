import hashlib
from pybloom_live import BloomFilter

class BaselineDHT:
    def __init__(self, peers):
        self.peers = peers
        self.keyword_map = {}
        self.failed_peers = set()

    def _hash(self, keyword):
        sha1 = hashlib.sha1(keyword.encode())
        return int(sha1.hexdigest(), 16) % len(self.peers)

    def publish(self, keyword, peer_id):
        responsible_peer_id = self._hash(keyword)
        if responsible_peer_id not in self.keyword_map:
            self.keyword_map[responsible_peer_id] = {}
        if keyword not in self.keyword_map[responsible_peer_id]:
            self.keyword_map[responsible_peer_id][keyword] = []
        if peer_id not in self.keyword_map[responsible_peer_id][keyword]:
            self.keyword_map[responsible_peer_id][keyword].append(peer_id)

    def lookup(self, keyword):
        responsible_peer_id = self._hash(keyword)
        if responsible_peer_id in self.failed_peers:
            return None, None
        return responsible_peer_id, self.keyword_map.get(responsible_peer_id, {}).get(keyword, [])

    def add_peer(self, peer):
        self.peers.append(peer)

    def remove_peer(self, peer):
        self.peers.remove(peer)
        for responsible_peer_id, keywords in self.keyword_map.items():
            for keyword, peer_ids in keywords.items():
                if peer.peer_id in peer_ids:
                    peer_ids.remove(peer.peer_id)

    def fail_peer(self, peer_id):
        self.failed_peers.add(peer_id)

    def recover_peer(self, peer_id):
        if peer_id in self.failed_peers:
            self.failed_peers.remove(peer_id)


class BloomFilterDHT:
    def __init__(self, peers, capacity=1000, error_rate=0.1):
        self.peers = peers
        self.keyword_map = {}
        self.capacity = capacity
        self.error_rate = error_rate
        self.failed_peers = set()

    def _hash(self, keyword):
        sha1 = hashlib.sha1(keyword.encode())
        return int(sha1.hexdigest(), 16) % len(self.peers)

    def publish(self, keyword, peer_id):
        responsible_peer_id = self._hash(keyword)
        if responsible_peer_id not in self.keyword_map:
            self.keyword_map[responsible_peer_id] = {}
        if keyword not in self.keyword_map[responsible_peer_id]:
            self.keyword_map[responsible_peer_id][keyword] = BloomFilter(
                capacity=self.capacity, error_rate=self.error_rate
            )
        self.keyword_map[responsible_peer_id][keyword].add(peer_id)

    def lookup(self, keyword):
        responsible_peer_id = self._hash(keyword)
        if responsible_peer_id in self.failed_peers:
            return None, None
        return responsible_peer_id, self.keyword_map.get(responsible_peer_id, {}).get(keyword)

    def add_peer(self, peer):
        self.peers.append(peer)

    def remove_peer(self, peer):
        self.peers.remove(peer)
        # for simplicity not re-hashing the keywords
        # just gonna remove the keywords published by the peer
        for responsible_peer_id, keywords in self.keyword_map.items():
            for keyword, bloom_filter in keywords.items():
                if peer.peer_id in bloom_filter:
                    # not ideal, as we can't remove items from a bloom filter
                    pass
    
    def fail_peer(self, peer_id):
        self.failed_peers.add(peer_id)

    def recover_peer(self, peer_id):
        if peer_id in self.failed_peers:
            self.failed_peers.remove(peer_id)
