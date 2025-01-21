import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mydatasets.base_dataset import BaseDataset
from agents.base_agent import Agent
import hydra
from tqdm import tqdm

@hydra.main(config_path="../config", config_name="change_rag_query", version_base="1.2")
def main(cfg):
    cfg.agent.agent = hydra.compose(config_name="agent/"+cfg.agent.agent, overrides=[]).agent
    cfg.agent.model = hydra.compose(config_name="model/"+cfg.agent.model, overrides=[]).model
    
    dataset = BaseDataset(cfg.dataset)
    agent = Agent(cfg.agent)
    samples = dataset.load_data(use_retreival=True)
    for sample in tqdm(samples):
        sample[cfg.question_key], _ = agent.predict(sample[cfg.dataset.question_key])
    dataset.dump_data(samples)
    
if __name__ == "__main__":
    main()