import yaml

with open("ppo.yaml") as f:
    args = yaml.load(f, Loader=yaml.FullLoader)

print(args)
