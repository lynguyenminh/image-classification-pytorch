import yaml


def load_config(config_file): 

    with open(config_file, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            exit(1)

    return config


def save_config(config, config_file):
    with open(config_file, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


