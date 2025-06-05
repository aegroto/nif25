import yaml
import json
import sys

from search import config_from_parameters

file = json.load(open(sys.argv[1], "r"))
parameters = file[sys.argv[2]][0]
# parameters = file[0]

config = config_from_parameters(parameters)
config_path = sys.argv[3]
yaml.safe_dump(config, open(config_path, "w"))

