class RandomAgent():
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self):
        return self.action_space.sample()