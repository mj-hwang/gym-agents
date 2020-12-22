class RandomAgent():
    def __init__(self, env):
        self.action_space = env.action_space

    def action(self):
        return self.action_space.sample()