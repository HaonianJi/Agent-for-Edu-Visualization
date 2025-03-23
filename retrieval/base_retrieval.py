from mydatasets.doc_dataset import DocDataset

class BaseRetrieval():
    def __init__(self, config):
        pass
    
    def prepare(self, dataset: DocDataset):
        pass
    
    def find_top_k(self, dataset: DocDataset):
        pass