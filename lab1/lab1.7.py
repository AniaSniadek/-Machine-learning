import numpy as np
import matplotlib.pyplot as plt
import math 

fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Wykres funkcji')

x = np.arange(-5,5,0.01)
y = 3 * x**2 + 4 * x - 2
g = math.e**x/(math.e**x + 1)

ax.plot(x, y, color='blue', lw=2)
ax.plot(x, g, color='red', lw=2)

plt.show()
