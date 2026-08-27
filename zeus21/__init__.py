from .bursty_sfh import *
from .constants import *
from .correlations import *
from .cosmology import *
from .inputs import *
from .LFs import *
from .maps import *
from .reionization import *
from .SED import *
from .sfrd import *
from .T21coefficients import * 
from .wrappers import *
from .z21_utilities import *

import warnings
warnings.filterwarnings("ignore", category=UserWarning) #to silence unnecessary warning in mcfit

from pathlib import Path

HERE = Path(__file__).resolve().parent

_ns = {"__file__": str(HERE / "zeus21" / "_version.py")}
exec((HERE / "zeus21" / "_version.py").read_text(), _ns)
VERSION = _ns["get_version"]()

print('zeus21 version ' + VERSION)