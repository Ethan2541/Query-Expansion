from gensim.models import Word2Vec
import numpy as np

data = None
model = Word2Vec(sentences=data, vector_size=200, window=10, sample=1e-3, min_count=0, sg=1)
model.save("entity_word_embeddings.model")

def tree_embedding(headings):
    # Load the model
    model = Word2Vec.load("entity_word_embeddings.model")
    # Get the embeddings for all the headings
    embeddings = []
    for heading in headings:
        heading_words = heading.split()
        heading_embedding = np.zeros(200)
        for word in heading_words:
            try:
                heading_embedding += model.wv[word]
            except KeyError:
                pass
        embeddings.append(heading_embedding)
    return np.mean(embeddings, axis=0)