import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from lab02.linalg_interp import cubic_spline

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def load_data(file_path):
    """
    Loads data from a file, skipping commented lines and detecting delimiters.
    """
    with open(file_path) as f:
        for line in f:
            if not line.startswith("#"):
                delimiter = ',' if ',' in line else '\t'
                break
    return np.loadtxt(file_path, delimiter=delimiter, comments="#")


def generate_subplots(xd, yd, title_prefix, output_file):
    """
    Generates a 3×2 grid of plots for linear, quadratic, and cubic splines for both datasets.
    """
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    spline_orders = [1, 2, 3]
    datasets = [("Water", xd[0], yd[0]), ("Air", xd[1], yd[1])]

    for i, (dataset_name, x_data, y_data) in enumerate(datasets):
        for j, order in enumerate(spline_orders):
            x_new = np.linspace(x_data[0], x_data[-1], 100)
            if order == 1:
                spline_func = lambda x: np.interp(x, x_data, y_data)  # Linear spline
            elif order == 2:
                spline_func = cubic_spline(x_data, y_data, order=2)  # Quadratic spline
            elif order == 3:
                spline_func = cubic_spline(x_data, y_data, order=3)  # Cubic spline

            y_new = spline_func(x_new)
            ax = axes[j, i]
            ax.scatter(x_data, y_data, color='red', label='Data points')
            ax.plot(x_new, y_new, color='blue', label=f'Order {order} Spline')
            ax.set_title(f"{title_prefix} ({dataset_name}) - Order {order}")
            ax.set_xlabel("Temperature (°C)")
            ax.set_ylabel("Density")
            ax.legend()
            ax.grid()

    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()
    print(f"Grid of plots saved to {output_file}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, "data")
    output_dir = os.path.join(base_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    water_data = load_data(os.path.join(data_dir, "water_density_vs_temp_usgs.txt"))
    air_data = load_data(os.path.join(data_dir, "air_density_vs_temp_eng_toolbox.txt"))

    # Extract x (temperature) and y (density) values
    xd_water, yd_water = water_data[:, 0], water_data[:, 1]
    xd_air, yd_air = air_data[:, 0], air_data[:, 1]

    # Generate and save a 3×2 grid of plots
    generate_subplots(
        xd=[xd_water, xd_air],
        yd=[yd_water, yd_air],
        title_prefix="Density vs. Temperature",
        output_file=os.path.join(output_dir, "density_splines_grid.png")
    )
