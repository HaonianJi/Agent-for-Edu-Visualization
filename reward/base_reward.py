class DummyRewardFunction:
    def __init__(self, config, **kwargs):
        self.config = config
        self.__name__ = "DummyRewardFunction"
        
    def __call__(self, completions, **kwargs):
        return [float(len(set(completion))) for completion in completions]