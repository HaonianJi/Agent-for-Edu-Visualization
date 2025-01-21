from agents.base_agent import Agent
from mydatasets.base_dataset import BaseDataset
from tqdm import tqdm
import importlib
import json
import torch
from typing import List

class MultiAgentSystem:
    def __init__(self, config):
        self.config = config
        self.agents:List[Agent] = []
        self.models:dict = {}
        for agent_config in self.config.agents:
            if agent_config.model.class_name not in self.models:
                module = importlib.import_module(agent_config.model.module_name)
                model_class = getattr(module, agent_config.model.class_name)
                print("Create model: ", agent_config.model.class_name)
                self.models[agent_config.model.class_name] = model_class(agent_config.model)
            self.add_agent(agent_config, self.models[agent_config.model.class_name])
            
        if config.sum_agent.model.class_name not in self.models:
            module = importlib.import_module(config.sum_agent.model.module_name)
            model_class = getattr(module, config.sum_agent.model.class_name)
            self.models[config.sum_agent.model.class_name] = model_class(config.sum_agent.model)
        self.sum_agent = Agent(config.sum_agent, self.models[config.sum_agent.model.class_name])
        
    def add_agent(self, agent_config, model):
        module = importlib.import_module(agent_config.agent.module_name)
        agent_class = getattr(module, agent_config.agent.class_name)
        agent:Agent = agent_class(agent_config, model)
        self.agents.append(agent)
        
    def method1(self, question, texts, images):
        round0_outputs, _ = self.execute(question, texts, images)
        round0_sr = self.selfreflect()
        
        original_discussion = {}

        for idx, output in enumerate(round0_outputs):
            original_discussion[idx] = f"Agent {idx}:\n" + output + "\n" + round0_sr[idx]
        
        discussion_log, agent_discussions = self.discuss(original_discussion, rounds = self.config.rounds)
        
        roundk_outputs, _ = self.execute(self.config.roundk_prompt, agent_discussions, None, discuss_mode=True)
        roundk_sr = self.selfreflect()
        
        discussion_log += "Conclusions after discussion:\n"
        for idx, output in enumerate(roundk_outputs):
            discussion_log += f"Agent {idx}:\n" + output + "\n" + roundk_sr[idx]
        
        final_ans, reasoning_ans, final_messages = self.sum(discussion_log)
        
        return final_ans, reasoning_ans, final_messages
        
    def execute(self, question, texts, images, discuss_mode=False):
        all_outputs = []
        all_messages = []
        
        for agent_idx, agent in enumerate(self.agents):
            if discuss_mode:
                outputs, messages = agent.predict(question, [texts[agent_idx]], images, with_sys_prompt=False)
            else:
                outputs, messages = agent.predict(question, texts, images, with_sys_prompt=True)
            all_outputs.append(outputs)
            all_messages.append(messages)
            
        return all_outputs, all_messages
        
    def selfreflect(self):
        all_outputs = []
        for agent in self.agents:
            outputs = agent.self_reflect()
            all_outputs.append(outputs)
        return all_outputs
            
    def discuss(self, original_discussion, rounds):        
        discussion_log = "Answers from agents:\n"
        for k, v in original_discussion.items():
            discussion_log += v
            
        discussion_log += "Discussion:\n"
        agent_discussions = {}
        for agent_idx in range(len(self.agents)):
            agent_discussions[agent_idx] = ""
            for k, v in original_discussion.items():
                if k != agent_idx:
                    agent_discussions[agent_idx] += v + "\n"
                    
        for round_idx in range(rounds):
            for agent_idx, agent in enumerate(self.agents):
                turn_discussion = agent.discuss(agent_discussions[agent_idx], agent_idx, agent_ids = list(range(len(self.agents))))
                for k, v in turn_discussion.items():
                    if v:
                        agent_discussions[k] += f"Agent {agent_idx} -> Agent {k}:\n" + v + "\n"
                        discussion_log += f"Agent {agent_idx} -> Agent {k}:\n" + v + "\n"
                        
        return discussion_log, agent_discussions
        
    def predict(self, question, texts, images):
        all_outputs, _ = self.execute(question, texts, images)
        sum_question = question + '\n'
        idx = 0
        for output in all_outputs:
            idx += 1
            sum_question += f"Answer {idx}: {output}"
            
        final_ans, all_messages = self.sum_agent.predict(sum_question, texts, images)
        return final_ans, all_messages
    
    def sum(self, sum_question):
        ans, all_messages = self.sum_agent.predict(sum_question)
        def extract_final_answer(agent_response):
            try:
                response_dict = json.loads(agent_response)
                return response_dict.get("Answer", None)
            except json.JSONDecodeError:
                return agent_response
        final_ans = extract_final_answer(ans)
        return final_ans, ans, all_messages

    def predict_dataset(self, dataset:BaseDataset):
        samples = dataset.load_data(use_retreival=True)
        if self.config.truncate_len:
            samples = samples[:self.config.truncate_len]
        for sample in tqdm(samples):
            question, texts, images = dataset.load_sample_retrieval_data(sample)
            try:
                final_ans, all_messages = self.predict(question, texts, images)
            except Exception as e:
                print(e)
                final_ans, all_messages = None, None
            sample[self.config.ans_key] = final_ans
            if self.config.save_message:
                sample[self.config.ans_key+"_message"] = all_messages
        path = dataset.dump_reults(samples)
        print(f"Save results to {path}.")
        
    def method1_dataset(self, dataset:BaseDataset):
        samples = dataset.load_data(use_retreival=True)
        if self.config.truncate_len:
            samples = samples[:self.config.truncate_len]
            
        sample_no = 0
        for sample in tqdm(samples):
            question, texts, images = dataset.load_sample_retrieval_data(sample)
            try:
                final_ans, reasoning_ans, final_messages = self.method1(question, texts, images)
            except RuntimeError as e:
                print(e)
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                final_ans, reasoning_ans, final_messages = None, None, None
            sample[self.config.ans_key] = final_ans
            if self.config.save_message:
                sample[self.config.ans_key+"_message"] = final_messages
            torch.cuda.empty_cache()
            for agent in self.agents:
                agent.clean_messages()
            self.sum_agent.clean_messages()
            
            sample_no += 1
            if sample_no % self.config.save_freq == 0:
                path = dataset.dump_reults(samples)
                print(f"Save {sample_no} results to {path}.")
        path = dataset.dump_reults(samples)
        print(f"Save final results to {path}.")
        

        


