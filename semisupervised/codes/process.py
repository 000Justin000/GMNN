import sys
import numpy as np

acc = np.loadtxt(sys.argv[1])

print('{:7.3f} ± {:7.3f}'.format(np.mean(acc), np.std(acc)))
