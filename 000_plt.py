import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


fig1, ax1 = plt.subplots() #figure 1
t1 = np.arange(0.0, 2.0, 0.1)
t2 = np.arange(0.0, 2.0, 0.01)

# note that plot returns a list of lines.  The "l1, = plot" usage
# extracts the first element of the list into l1 using tuple
# unpacking.  So l1 is a Line2D instance, not a sequence of lines
l1, = ax1.plot(t2, np.exp(-t2))
l2, l3 = ax1.plot(t2, np.sin(2 * np.pi * t2), '--o', t1, np.log(1 + t1), '.')
l4, = ax1.plot(t2, np.exp(-t2) * np.sin(2 * np.pi * t2), 's-.')

ax1.legend((l2, l4), ('oscillatory', 'damped'), loc='upper right', shadow=True)
ax1.set_xlabel('time')
ax1.set_ylabel('volts')
ax1.set_title('Damped oscillation')



fig2, (ax2_1, ax2_2) = plt.subplots(2, 1)   #figure 2
x = np.linspace(0, 1)

# Plot the lines y=x**n for n=1..4.
for n in range(1, 5):
    ax2_1.plot(x, x**n, label="n={0}".format(n))
leg = ax2_1.legend(loc="upper left", bbox_to_anchor=[0, 1],
                 ncol=2, shadow=True, title="Legend", fancybox=True)
leg.get_title().set_color("red")

# Demonstrate some more complex labels.
ax2_2.plot(x, x**2, label="multi\nline")
half_pi = np.linspace(0, np.pi / 2)
ax2_2.plot(np.sin(half_pi), np.cos(half_pi), label=r"$\frac{1}{2}\pi$")
ax2_2.plot(x, 2**(x**2), label="$2^{x^2}$")
ax2_2.legend(shadow=True, fancybox=True)

plt.show()


