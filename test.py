import math
RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"

import numpy as np

WidthChart=[676,655,633,612,591,569,548,527,505,484,463,441,420,399,377,356,335,313,292,271,249,228,207,185,164,143,121,100,79,57,36,15]
WidthChart.reverse()
HeightChart=[10,24,38,52,67,81,95,109,124,138,152,166,180,195,209,223,237,252,266,280,294,308,323,337,351,365,380,394,408,422,436,451]
og = HeightChart

x = np.arange(1, 33)
y = np.array(og)

a, b = np.polyfit(x, y, 1)
print("a =", a, "b =", b)

for x in range(32):
    print((round(a*(x+1)+b)==og[x]))