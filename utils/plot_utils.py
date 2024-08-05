import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import statsmodels.api as sm

# Generate some example time series data
np.random.seed(42)
n_points = 100
x = np.linspace(0, 10, n_points)
y = 3 * x + np.random.normal(size=n_points)

# Set up the figure, the axis, and the plot elements
fig, ax = plt.subplots()
ax.set_xlim((0, 10))
ax.set_ylim((np.min(y), np.max(y)))
(line,) = ax.plot([], [], lw=2, label="OLS Prediction")
(data_line,) = ax.plot([], [], lw=1, label="Data")


def init():
    line.set_data([], [])
    data_line.set_data([], [])
    return line, data_line


def animate(i):
    if i == 0:
        return line, data_line

    x_subset = x[:i]
    y_subset = y[:i]
    X = sm.add_constant(x_subset)
    model = sm.OLS(y_subset, X).fit()
    y_pred = model.predict(sm.add_constant(x))

    line.set_data(x, y_pred)
    data_line.set_data(x_subset, y_subset)
    return line, data_line


ani = animation.FuncAnimation(
    fig, animate, init_func=init, frames=n_points, interval=100, blit=True
)

# Add legend
ax.legend()

# Save as gif (requires imagemagick)
ani.save("ols_evolution.gif", writer="imagemagick")

# Display the animation
plt.show()
