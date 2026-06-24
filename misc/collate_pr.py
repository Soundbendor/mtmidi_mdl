import os
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from util import util_main as UMN
from util import util_constants as UC
from util import util_rdb as UR
from util import util_optuna as UO
import sys

models = ["musicgen-small", "musicgen-medium", "musicgen-large", "MERT-v1-95M", "MERT-v1-330M", "wav2vec2-base", "wav2vec2-large"]
