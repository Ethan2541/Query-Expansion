# Query-Expansion

Degraded implementation of the research paper "Local and Global Query Expansion for Hierarchical Complex Topics" written by Dalton, Naseri, Dietz, and Allan. It is part of the first year course [Information Retrieval and Natural Language Processing](https://dac.lip6.fr/master/enseignement/rital/) from Sorbonne Université's computer science master.


## Expected Features

- Use of the TREC CAR dataset (2018) for evaluation
- Word-based retrieval baselines and comparisons (experimental results, t-tests, ...)
- Tree structure T for hierarchical topics, and document representation D
- Entity E, word representations W, aliases A
- Local expansion: probability of relevance for a topic and its subtopics
- Global expansion: joint entity-word embeddings
- Features combination with a log-linear model