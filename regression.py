import numpy as np
import matplotlib.pyplot as plt

dist = np.arange(2, 28, 1)
disp = np.array([215, 188, 174, 164, 159, 155, 152, 150, 152, 146, 
                145, 143, 144, 141, 141, 141, 140, 140, 140, 139, 139, 139, 142, 138, 139, 137])

# x = np.arange(137, 216, 1)
# y = np.exp(23.08) * (1.2442 ** -x) * (1.006 ** (x ** 2))

# polynomial fit
model = np.poly1d(np.polyfit(disp, dist, 6))
polyline = np.linspace(137, 188, 400)

# exponential fit
# fit = np.polyfit(disp, np.log(dist), 2)
# print(fit)

plt.figure()
plt.scatter(disp, dist, label='Scatter')
# plt.plot(x, y)
plt.plot(polyline, model(polyline), color='red', label=str(model))
plt.xlabel('Disparity (px)')
plt.ylabel('Distance (m)')
plt.title('Distance vs Disparity (Regression Line)')
plt.legend()
plt.show()
