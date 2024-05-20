# This file should implement the required baselines for model comparisons, namely:
# L2R-SDM-RM3

import numpy as np

class RIH_QL(object):
    def __init__(self, vectorizer):
        self.tokenizer = vectorizer.build_tokenizer()
        self.vectorizer = vectorizer

    def get_query(self, query):
        return " ".join([query[2], *query[3], query[1]])

    def get_query_likelihood(self, query, doc):
        q = self.vectorizer.transform([self.get_query(query)]).toarray()[0]
        d = self.vectorizer.transform([doc]).toarray()[0]
        dl = len(self.tokenizer(doc))
        if dl == 0:
            score = -np.inf
        else:
            score = np.where(q > 0, np.log(d/dl + 1e-12), np.log(1-d/dl + 1e-12)).sum()
        return score

    def get_top_k(self, query, docs, k=1000):
        documents_ids, scores = [], []
        for doc_id, doc in docs:
            documents_ids.append(doc_id)
            scores.append(self.get_query_likelihood(query, doc))
        documents_ids, scores = np.array(documents_ids), np.array(scores)
        top_k_indexes = np.argsort(scores)[::-1][:k]
        return documents_ids[top_k_indexes], scores[top_k_indexes]
    

class RIH_Cosine(object):
    def __init__(self, documents_ids, documents_vectors, vectorizer):
        self.documents_ids = documents_ids
        self.documents_vectors = documents_vectors
        self.vectorizer = vectorizer

    def get_query(self, query):
        return " ".join([query[1], *query[2], query[0]])

    def cosine_similarities(self, query):
        dot_product = self.documents_vectors @ query.reshape(-1, 1)
        denominator = (np.maximum(np.linalg.norm(query), 1e-12) * np.maximum(np.linalg.norm(self.documents_vectors), 1e-12))
        return dot_product / denominator

    def get_top_k(self, query, k=1000):
        q = self.vectorizer.transform([self.get_query(query)]).toarray()[0]
        scores = self.cosine_similarities(q).reshape(-1)
        top_k_indexes = np.argsort(scores)[::-1][:k]
        return self.documents_ids[top_k_indexes], scores[top_k_indexes]