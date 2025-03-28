from agents.multi_agent_system import MultiAgentSystem

class Method1(MultiAgentSystem):
    def predict(self, question, texts=None, images=None):
        """
        an example of how to custom the framework
        """
        
        # all agents predict the answer
        round0_outputs, _ = self.execute(question, texts, images)
        
        # all agents self-reflect
        round0_sr = self.selfreflect()
        
        original_discussion = {}

        # prepare the original discussion data: agent's answer + self-reflection
        for idx, output in enumerate(round0_outputs):
            original_discussion[idx] = f"Agent {idx}:\n" + output + "\n" + round0_sr[idx]
        
        # start discussion
        discussion_log, agent_discussions = self.discuss(original_discussion, rounds = self.config.rounds)
        
        # all agents predict the answer and self-reflect after discussion
        roundk_outputs, _ = self.execute(self.config.roundk_prompt, agent_discussions, None, discuss_mode=True)
        roundk_sr = self.selfreflect()
        
        # complete the discussion log
        discussion_log += "Conclusions after discussion:\n"
        for idx, output in enumerate(roundk_outputs):
            discussion_log += f"Agent {idx}:\n" + output + "\n" + roundk_sr[idx]
        
        # sum the discussion log
        final_ans, final_messages = self.sum(discussion_log)
        
        # clean the history messages for all agents
        self.clean_messages()
        
        return final_ans, final_messages
    

class Method2(MultiAgentSystem):
    def predict(self, question, texts=None, images=None):
        """
        an example of how to custom the framework
        """
        
        # all agents predict the answer
        outputs, messages = self.execute(question, texts, images)
        
        discussion = ""

        # prepare the original discussion data: agent's answer + self-reflection
        for idx, output in enumerate(outputs):
            discussion += f"Agent {idx}:\n" + output
        
        self.clean_messages()
            
        return discussion, messages