import os, sys
parentddir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parentddir)

from dotenv import load_dotenv
load_dotenv()

RANDOM_SEED = 42 # life, the universe and everything” is 42!