class RetrievalResults:
    def __init__(self, results: list[dict]):
        self._results = results

    def __iter__(self):
        return iter(self._results)

    def __getitem__(self, idx):
        return self._results[idx]

    def __len__(self):
        return len(self._results)

    @property
    def chunk_text(self):
        return [r["chunk_text"] for r in self._results]

    @property
    def source(self):
        return [r["source"] for r in self._results]

    @property
    def similarity(self):
        return [r["similarity"] for r in self._results]
