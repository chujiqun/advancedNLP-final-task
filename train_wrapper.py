import json
import shutil
import sys
from time import time
import logging

from allennlp.commands import main
import warnings
warnings.filterwarnings("ignore")

# 目前支持: [bimpm, cafe, decomposable_attention, esim, match_pyramid, mv_lstm, san]

# model = "match_pyramid"
model = "mv_lstm"
config_file = "config/%s.jsonnet" % model

overrides = json.dumps({"trainer": {"cuda_device": 1}})

serialization_dir = "checkpoint/%s" % model

shutil.rmtree(serialization_dir, ignore_errors=True)
logger = logging.getLogger(__name__)
# Assemble the command into sys.argv
sys.argv = [
    "allennlp",  # command name, not used by main
    "train",
    config_file,
    "-s", serialization_dir,
    "--include-package", "library",
    "-o", overrides,
]
begin = time()

main()

consume = ( time() - begin ) / 60
print( "consum %.2f min" % consume )
