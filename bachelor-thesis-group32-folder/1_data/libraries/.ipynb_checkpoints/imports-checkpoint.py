import fitz
import re
from collections import defaultdict
import pandas as pd
import unicodedata
from itertools import product
import os
from tqdm import tqdm

import camelot
import tabula