import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mydatasets.doc_dataset import DocDataset
import hydra

@hydra.main(config_path="../config", config_name="base_doc", version_base="1.2")
def main(cfg):
    dataset = DocDataset(cfg.doc_dataset)
    dataset.extract_content()

if __name__ == "__main__":
    main()