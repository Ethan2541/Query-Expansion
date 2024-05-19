# This file should implement the required baselines for model comparisons, namely:
# L2R-SDM-RM3

import numpy as np

class RIH_QL(object):
    def __init__(self, index, vectorizer):
        self.index = index
        self.vectorizer = vectorizer

    def get_query(self, query):
        return " ".join([query[2], *query[3], query[1]])

    def get_query_likelihood(self, query, doc):
        q = self.vectorizer.transform([self.get_query(query)]).toarray()[0]
        d = self.index[doc[0]]["vector"]
        dl = self.index[doc[0]]["n_tokens"]
        score = np.where(q > 0, np.log(d/dl), np.log(1-d/dl)).sum()
        return score

    def get_top_k(self, query, docs, k=1000):
        scores = {}
        for doc_id, _ in docs:
            scores[doc_id] = self.get_query_likelihood(query, doc_id)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    

class RIH_Cosine(object):
    def __init__(self, documents_ids, documents_vectors, vectorizer):
        self.documents_ids = documents_ids
        self.documents_vectors = documents_vectors
        self.vectorizer = vectorizer

    def get_query(self, query):
        return " ".join([query[2], *query[3], query[1]])

    def cosine_similarities(self, query):
        vectors_product = query * self.documents_vectors
        return vectors_product / (np.maximum(np.linalg.norm(query), 1e-12) * np.maximum(np.linalg.norm(self.documents_vectors, axis=1), 1e-12)) * vectors_product.sum(axis=1)

    def get_top_k(self, query, k=1000):
        q = self.vectorizer.transform([self.get_query(query)]).toarray()[0]
        scores = self.cosine_similarities(q)
        top_k_indexes = np.argsort(scores)[::-1][:k]
        return self.documents_ids[top_k_indexes], scores[top_k_indexes]