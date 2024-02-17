import yaml

with open("config.yaml") as f:
    args = yaml.load(f, Loader=yaml.FullLoader)

print(args)
