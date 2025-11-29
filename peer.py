from collections import OrderedDict

class Peer:
    def __init__(self, peer_id, cache_size=10):
        self.peer_id = peer_id
        self.documents = []
        self.index = {}
        self.cache = OrderedDict()
        self.cache_size = cache_size

    def add_document(self, doc):
        self.documents.append(doc)

    def create_index(self):
        for doc in self.documents:
            for keyword in doc.split():
                if keyword not in self.index:
                    self.index[keyword] = []
                self.index[keyword].append(doc)

    def query(self, keyword):
        return self.index.get(keyword, [])

    def get_from_cache(self, keyword):
        if keyword in self.cache:
            # move tto end to show recent use
            self.cache.move_to_end(keyword)
            return self.cache[keyword]
        return None

    def add_to_cache(self, keyword, value):
        if len(self.cache) >= self.cache_size:
            # get rid of least recently used item
            self.cache.popitem(last=False)
        self.cache[keyword] = value
