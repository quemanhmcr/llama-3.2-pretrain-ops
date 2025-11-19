from typing import List, Set
try:
    from datasketch import MinHash, MinHashLSH
except ImportError:
    # logging.error("pip install datasketch")
    pass

class MinHashDeduplicator:
    __slots__ = ('lsh', 'num_perm', 'threshold', 'ngram_size')
    def __init__(self, threshold: float = 0.8, num_perm: int = 128, ngram_size: int = 3):
        self.threshold = threshold
        self.num_perm = num_perm
        self.ngram_size = ngram_size
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    
    def _create_shingles(self, text: str) -> Set[str]:
        tokens = text.lower().split()
        if len(tokens) < self.ngram_size: return {' '.join(tokens)}
        shingles = set()
        for i in range(len(tokens) - self.ngram_size + 1):
            shingle = ' '.join(tokens[i:i + self.ngram_size])
            shingles.add(shingle)
        return shingles
    
    def create_minhash(self, text: str) -> MinHash:
        minhash = MinHash(num_perm=self.num_perm)
        shingles = self._create_shingles(text)
        for shingle in shingles: minhash.update(shingle.encode('utf8'))
        return minhash
    
    def query(self, minhash: MinHash) -> List[str]: return self.lsh.query(minhash)
    def insert(self, doc_id: str, minhash: MinHash): self.lsh.insert(doc_id, minhash)
