import os

# it raises an error if not found
CODE_PATH = os.environ['SURVEY_CODE'] if 'SURVEY_CODE' in os.environ else os.path.join(os.environ['HOME'], 'github', 'jingwei')
ROOT_PATH = os.environ['SURVEY_DATA'] if 'SURVEY_DATA' in os.environ else os.path.join(os.environ['HOME'], 'VisualSearch')
OUTPUT_PATH = os.environ['SURVEY_DB'] if 'SURVEY_DB' in os.environ else os.path.join(os.environ['HOME'], 'VisualSearch')
MATLAB_PATH = os.environ['MATLAB_PATH'] if 'MATLAB_PATH' in os.environ else '/usr/local/'

DEFAULT_TPP = 'lemm'
DEFAULT_NEG_FILTER = 'co'
DEFAULT_POS_NR = 500

