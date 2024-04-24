# Fetch DBpedia dump
# Train the embeddings on DBpedia dump
# For each heading, add the 3 most similar embeddings (before or after?). Same goes for the tree

from gensim.models import Word2Vec
from SPARQLWrapper import SPARQLWrapper, JSON
import numpy as np


sparql = SPARQLWrapper("http://dbpedia.org/sparql")
sparql.setReturnFormat(JSON)

sparql.setQuery("""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX dcat: <http://www.w3.org/ns/dcat#>
    PREFIX dct: <http://purl.org/dc/terms/>
    PREFIX dcv: <https://dataid.dbpedia.org/databus-cv#>
    PREFIX databus: <https://dataid.dbpedia.org/databus#>
    SELECT ?file WHERE
    {
        GRAPH ?g
        {
            ?dataset databus:artifact <https://databus.dbpedia.org/dbpedia/text/long-abstracts> .
            {
                ?distribution dct:hasVersion ?version {
                    SELECT (?v as ?version) { 
                        GRAPH ?g2 { 
                            ?dataset databus:artifact <https://databus.dbpedia.org/dbpedia/text/long-abstracts> . 
                            ?dataset dct:hasVersion ?v . 
                        }
                    } ORDER BY DESC (STR(?version)) LIMIT 1 
                }
            }
            ?dataset dcat:distribution ?distribution .
            ?distribution databus:file ?file .
        }
    }
""")


sparql.setReturnFormat(JSON)
results = sparql.query().convert()
print(results)

# data = None
# model = Word2Vec(sentences=data, vector_size=200, window=10, sample=1e-3, min_count=0, sg=1)
# model.save("entity_word_embeddings.model")

def global_embedding(headings):
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