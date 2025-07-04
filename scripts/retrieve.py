import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mydatasets.doc_dataset import DocDataset
from retrieval.base_retrieval import BaseRetrieval
import hydra
import importlib

@hydra.main(config_path="../config", config_name="base_doc", version_base="1.2")
def main(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.retrieval.cuda_visible_devices
    retrieval_class_path = cfg.retrieval.class_path
    module_name, class_name = retrieval_class_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    retrieval_class = getattr(module, class_name)
    
    dataset = DocDataset(cfg.doc_dataset)
    retrieval_model:BaseRetrieval = retrieval_class(cfg.retrieval)
    retrieval_model.find_top_k(dataset, prepare=True)

if __name__ == "__main__":
    main()