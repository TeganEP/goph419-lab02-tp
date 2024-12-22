import sys
import os

# Ensure the project root is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from lab02.linalg_interp import cubic_spline


def load_data(file_path):
    """
    Loads data from a file, skipping commented lines and detecting delimiters.

    Parameters:
        file_path (str): Path to the data file.

    Returns:
        numpy.ndarray: Data array loaded from the file.
    """
    try:
        with open(file_path) as f:
            for line in f:
                if not line.startswith("#"):  # Skip commented lines
                    delimiter = ',' if ',' in line else '\t'
                    break
        return np.loadtxt(file_path, delimiter=delimiter, comments="#")
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        raise


def generate_spline_plot(xd, yd, title, output_file):
    """
    Generates a cubic spline plot for the given data and saves it to a file.

    Parameters:
        xd (array): Independent variable (x-axis data).
        yd (array): Dependent variable (y-axis data).
        title (str): Title for the plot.
        output_file (str): Path to save the plot.
    """
    x_new = np.linspace(xd[0], xd[-1], 100)
    spline_func = cubic_spline(xd, yd)
    y_new = spline_func(x_new)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(xd, yd, color='red', label='Data points')
    plt.plot(x_new, y_new, label="Cubic Spline", color='blue')
    plt.title(title, fontsize=14)
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()
    print(f"Plot saved to {output_file}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))  # Go to the root of the project
    data_dir = os.path.join(base_dir, "data")  # Correctly locate the "data" folder at the root
    output_dir = os.path.join(base_dir, "figures")  # Save figures in the "figures" folder
    os.makedirs(output_dir, exist_ok=True)

    # Load data files
    water_data = load_data(os.path.join(data_dir, "water_density_vs_temp_usgs.txt"))
    air_data = load_data(os.path.join(data_dir, "air_density_vs_temp_eng_toolbox.txt"))

    # Extract x (temperature) and y (density) values
    xd_water, yd_water = water_data[:, 0], water_data[:, 1]
    xd_air, yd_air = air_data[:, 0], air_data[:, 1]

    # Generate and save plots
    generate_spline_plot(xd_water, yd_water, "Water Density", os.path.join(output_dir, "water_density_splines.png"))
    generate_spline_plot(xd_air, yd_air, "Air Density", os.path.join(output_dir, "air_density_splines.png"))
