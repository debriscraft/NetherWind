"""Public launcher: forwards CLI args to the compiled trainer."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.train_mappo import main
if __name__ == '__main__':
    main()
