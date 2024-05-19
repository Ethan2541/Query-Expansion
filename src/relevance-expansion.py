import numpy as np

def rocchio(query, relevant_docs, non_relevant_docs, alpha=1.0, beta=0.75, gamma=0.15):
    expanded_query = alpha*query + beta*relevant_docs.mean(axis=0) - gamma*non_relevant_docs.mean(axis=0)
    return np.maximum(expanded_query, 0)