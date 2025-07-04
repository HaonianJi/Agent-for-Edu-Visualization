import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mydatasets.doc_dataset import DocDataset
from agents.mdoc_agent import MDocAgent
import hydra

@hydra.main(config_path="../config", config_name="base_doc", version_base="1.2")
def main(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.multi_agents.cuda_visible_devices
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
    for agent_config in cfg.multi_agents.agents:
        agent_name = agent_config.agent
        model_name = agent_config.model
        agent_cfg = hydra.compose(config_name="agent/"+agent_name, overrides=[]).agent
        model_cfg = hydra.compose(config_name="model/"+model_name, overrides=[]).model
        agent_config.agent = agent_cfg
        agent_config.model = model_cfg
    
    cfg.multi_agents.sum_agent.agent = hydra.compose(config_name="agent/"+cfg.multi_agents.sum_agent.agent, overrides=[]).agent
    cfg.multi_agents.sum_agent.model = hydra.compose(config_name="model/"+cfg.multi_agents.sum_agent.model, overrides=[]).model
    
    dataset = DocDataset(cfg.doc_dataset)
    multi_agents = MDocAgent(cfg.multi_agents)
    multi_agents.predict_dataset(dataset)
    
if __name__ == "__main__":
    main()