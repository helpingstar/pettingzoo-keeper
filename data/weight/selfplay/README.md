## v1

```python
class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(35, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(35, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 18), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
```