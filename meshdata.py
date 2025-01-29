import numpy as np
import matplotlib.pyplot as plt

# Load the points from your .dat file (adjust based on the file format)
points_2d = np.loadtxt('mesh.dat', skiprows=1)  # Adjust based on your file format

# If the data is only X and Y, you can plot them directly
x = points_2d[:, 0]
y = points_2d[:, 1]

# If you want to visualize it in a scatter plot
plt.scatter(x, y, s=50, color='blue')  # 's=1' is for point size, you can adjust

# Optional: Set axis limits if needed
plt.xlim([min(x) - 1, max(x) + 1])
plt.ylim([min(y) - 1, max(y) + 1])

# Set labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('2D Point Cloud')

# Show the plot
plt.show()
plt.savefig('point_cloud_data.png', bbox_inches='tight')