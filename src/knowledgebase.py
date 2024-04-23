class KnowledgeBase(object):
    def __init__(self):
        self.kb = set()

    def build_knowledge_base(self, dataset):
        for query in dataset:
            # The title of the article is the entity
            entity = query[2]
            self.kb.add(entity)

    def save_knowledge_base(self, filename):
        with open(filename, 'w') as f:
            for entity in self.kb:
                f.write(entity + '\n')

    def load_knowledge_base(self, filename):
        with open(filename, 'r') as f:
            for line in f:
                self.kb.add(line.strip())