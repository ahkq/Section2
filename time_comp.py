import numpy as np
import matplotlib.pyplot as plt
import time
from convex_hull import graham_scan, jarvis_march, quickhull, monotone_chain

def generate_point_cloud(n):
    points = np.random.rand(n,2)
    return points

def measure_runtime():
    n_values = [10, 50, 100, 200, 400, 800, 1000]
    graham_times = []
    jarvis_times = []
    quickhull_times = []
    monotone_times = []

    for n in n_values:
        points = generate_point_cloud(n)
        
        # Measure Graham Scan time
        print(f"Running Graham Scan for n={n} points")
        start_time = time.time()
        graham_scan(points)
        graham_times.append(time.time() - start_time)

        # Measure Jarvis March time
        print(f"Running Jarvis March for n={n} points")
        start_time = time.time()
        jarvis_march(points)
        jarvis_times.append(time.time() - start_time)

        # Measure QuickHull time
        print(f"Running QuickHull for n={n} points")
        start_time = time.time()
        quickhull(points)
        quickhull_times.append(time.time() - start_time)

        # Measure Monotone Chain time
        print(f"Running Monotone for n={n} points")
        start_time = time.time()
        monotone_chain(points)
        monotone_times.append(time.time() - start_time)

measure_runtime()