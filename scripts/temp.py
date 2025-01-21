import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mydatasets.base_dataset import BaseDataset
from agents.multi_agent_system import MultiAgentSystem
import hydra

@hydra.main(config_path="../config", config_name="base", version_base="1.2")
def main(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.multi_agents.cuda_visible_devices
    
    for agent_config in cfg.multi_agents.agents:
        agent_name = agent_config.agent
        model_name = agent_config.model
        agent_cfg = hydra.compose(config_name="agent/"+agent_name, overrides=[]).agent
        model_cfg = hydra.compose(config_name="model/"+model_name, overrides=[]).model
        agent_config.agent = agent_cfg
        agent_config.model = model_cfg
    
    cfg.multi_agents.sum_agent.agent = hydra.compose(config_name="agent/"+cfg.multi_agents.sum_agent.agent, overrides=[]).agent
    cfg.multi_agents.sum_agent.model = hydra.compose(config_name="model/"+cfg.multi_agents.sum_agent.model, overrides=[]).model
    
    dataset = BaseDataset(cfg.dataset)
    sample = dataset.load_data()[0]
    question, texts, images = dataset.load_sample_retrieval_data(sample)
    multi_agents = MultiAgentSystem(cfg.multi_agents)
    final_ans, reasoning_ans, final_messages = multi_agents.method1(question, texts, images)
    print(final_ans)
    # print(reasoning_ans)
    print(final_messages)
    
if __name__ == "__main__":
    main()