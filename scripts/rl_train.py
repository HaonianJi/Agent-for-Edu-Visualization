import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets import load_dataset
import hydra
from agents.base_agent import Agent
from utils.config import name_to_class

@hydra.main(config_path="../config", config_name="base_train", version_base="1.2")
def main(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda_visible_devices
    dataset = load_dataset("trl-lib/tldr", split="train")
    
    if cfg.dataset.truncate_len:
        print(f"Truncating dataset to {cfg.dataset.truncate_len} examples")
        dataset = dataset.select(range(cfg.dataset.truncate_len))
        
    cfg.totrain_agent.agent = hydra.compose(config_name="agent/"+cfg.totrain_agent.agent, overrides=[]).agent
    cfg.totrain_agent.model = hydra.compose(config_name="model/"+cfg.totrain_agent.model, overrides=[]).model
    
    totrain_agent = Agent(cfg.totrain_agent)
    model = totrain_agent.model.model
    
    trainer = name_to_class(cfg.train)
    reward = name_to_class(cfg.reward)
    
    trainer.train(model, dataset, reward)
    
if __name__ == "__main__":
    main()