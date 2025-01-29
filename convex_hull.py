import numpy as np
import matplotlib.pyplot as plt

def read_points_from_file(file_name):
    points_2d = np.loadtxt(file_name, skiprows=1)  # Adjust based on your file format
    return points_2d

# Graham Scan
def graham_scan(points):
    points = sorted(points.tolist())  # Sort points lexicographically by x, then y
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return np.array(lower[:-1] + upper[:-1])

# Jarvis March
def jarvis_march(points):
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    n = len(points)
    if n < 3:
        return points  

    leftmost = min(points, key=lambda p: p[0])

    hull = []
    p = leftmost
    while True:
        hull.append(p)
        q = points[0]
        for i in range(1, n):
            if cross(p, q, points[i]) > 0:
                q = points[i]
        p = q
        if np.array_equal(p, leftmost):
            break

    return np.array(hull)

# QuickHull
def quickhull(points):
    def orientation(p, q, r):
        return (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])

    def farthest_point_func(points, p1, p2):
        max_dist = 0
        farthest_point = None
        for p in points:
            dist = np.abs((p2[1] - p1[1]) * (p[0] - p1[0]) - (p2[0] - p1[0]) * (p[1] - p1[1]))
            if dist > max_dist:
                max_dist = dist
                farthest_point = p
        return farthest_point

    def find_hull(points, p1, p2):
        left, right = [], []
        for p in points:
            if orientation(p1, p2, p) > 0:
                left.append(p)
            elif orientation(p1, p2, p) < 0:
                right.append(p)

        if not left and not right:
            return [p1, p2]

        farthest_point = farthest_point_func(left + right, p1, p2)
        left_hull = find_hull(left, p1, farthest_point)
        right_hull = find_hull(right, farthest_point, p2)

        return left_hull[:-1] + right_hull

    if len(points) < 3:
        return points

    leftmost = min(points, key=lambda p: p[0])
    rightmost = max(points, key=lambda p: p[0])

    hull = [leftmost] + find_hull(points, leftmost, rightmost) + [rightmost]
    return np.array(hull)

# Monotone Chain
def monotone_chain(points):
    points = sorted(points.tolist())

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return np.array(lower[:-1] + upper[:-1])

def plot_convex_hull(points, hull_algorithms):
    plt.figure(figsize=(8, 8))
    
    # Plot point cloud
    plt.scatter(points[:, 0], points[:, 1], color='gray', label='Point Cloud', alpha=0.5)

    colors = ['red', 'blue', 'green', 'purple']
    labels = ['Graham Scan', 'Jarvis March', 'QuickHull', 'Monotone Chain']
    
    for i, hull_algorithm in enumerate(hull_algorithms):
        hull = hull_algorithm(points)
        plt.plot(hull[:, 0], hull[:, 1], color=colors[i], label=labels[i])
        plt.fill(hull[:, 0], hull[:, 1], color=colors[i], alpha=0.3)  # Fill the convex hull area

    plt.legend()
    plt.title("Convex Hulls and Point Cloud")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid(True)
    plt.show()

def plot_convex_hull_with_point_cloud(points, hull_algorithms):
    colors = ['red', 'cyan', 'green', 'purple']  # Different colors for each hull
    algorithm_names = ['Graham Scan', 'Jarvis March', 'Quickhull', 'Monotone Chain']
    
    for i, hull_algorithm in enumerate(hull_algorithms):

        plt.figure(figsize=(8, 8))

        x = points[:, 0]
        y = points[:, 1]
        plt.scatter(x, y, s=50, color='blue', label='Point Cloud')

        hull = hull_algorithm(points)
        
        plt.plot(np.append(hull[:, 0], hull[0, 0]), np.append(hull[:, 1], hull[0, 1]), 
                 color=colors[i], label=algorithm_names[i])

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f"Convex Hull - {algorithm_names[i]}")
        plt.legend()
        plt.grid(True)

        # Save figures to PNG files
        plt.savefig(f"{algorithm_names[i].replace(' ', '_').lower()}.png", bbox_inches='tight')
        
        plt.close()

file_name = 'mesh.dat' 
points = read_points_from_file(file_name)  


hull_algorithms = [graham_scan, jarvis_march, quickhull, monotone_chain]

plot_convex_hull_with_point_cloud(points, hull_algorithms)