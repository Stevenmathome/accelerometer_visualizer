import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
from filterpy.kalman import KalmanFilter

# Function to read data from CSV
def read_csv(file_path):
    times, accelerations_x, accelerations_y, accelerations_z = [], [], [], []
    with open(file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)  # Skip the header row if present
        for row in csv_reader:
            time, accel_x, accel_y, accel_z = map(float, row)
            times.append(time)
            accelerations_x.append(accel_x)
            accelerations_y.append(accel_y)
            accelerations_z.append(accel_z)
    return times, accelerations_x, accelerations_y, accelerations_z

# Function to simulate 3D movement
def simulate_movement(times, accelerations_x, accelerations_y, accelerations_z):
    dt = times[1] - times[0]  # Assuming equally spaced time points, the chip DOES do this
    positions_x, positions_y, positions_z = [0], [0], [0]  # Initial position is (0, 0, 0)

    for i in range(1, len(times)):
        new_pos_x = positions_x[-1] + (accelerations_x[i] * dt ** 2) / 2
        new_pos_y = positions_y[-1] + (accelerations_y[i] * dt ** 2) / 2
        new_pos_z = positions_z[-1] + (accelerations_z[i] * dt ** 2) / 2
        positions_x.append(new_pos_x)
        positions_y.append(new_pos_y)
        positions_z.append(new_pos_z)

    return positions_x, positions_y, positions_z

# Colormap vs time
def create_time_colormap(times):
    time_min, time_max = min(times), max(times)
    norm = mcolors.Normalize(vmin=time_min, vmax=time_max)
    cmap = plt.get_cmap('rainbow')
    return cmap(norm(times))

# Function to update the animation at each frame - this could be wrong
def update(frame, positions_x, positions_y, positions_z, sc, colors, kalman_filters):
    for i, kalman_filter in enumerate(kalman_filters):
        # Predict the next state
        kalman_filter.predict()

        # Update the Kalman filter with the noisy measurements
        z = np.array([[positions_x[frame]], [positions_y[frame]], [positions_z[frame]]])
        kalman_filter.update(z[i])

        # Get the corrected state estimate from the Kalman filter
        x = kalman_filter.x[0, 0]
        y = kalman_filter.x[1, 0]
        z = kalman_filter.x[2, 0]

        # Update the scatter with all frames up to present
        sc._offsets3d = (positions_x[:frame+1], positions_y[:frame+1], positions_z[:frame+1])
        sc.set_color(colors[:frame+1])
        ax.auto_scale_xyz(positions_x, positions_y, positions_z)

    return sc,

if __name__ == '__main__':
    csv_file_path = 'sourcedata.csv' #change name for data

    times, accelerations_x, accelerations_y, accelerations_z = read_csv(csv_file_path)

    positions_x, positions_y, positions_z = simulate_movement(times, accelerations_x, accelerations_y, accelerations_z)

    # Colors based on time
    colors = create_time_colormap(times)

    # 3D and Scatter
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    sc = ax.scatter([], [], [], c=[], cmap='rainbow', s=50)

    # View limits
    ax.set_xlim(min(positions_x), max(positions_x))
    ax.set_ylim(min(positions_y), max(positions_y))
    ax.set_zlim(min(positions_z), max(positions_z))

    # Create Kalman filters for 3D position and velocity for x, y, and z dimensions
    kalman_filters = [KalmanFilter(dim_x=6, dim_z=1) for _ in range(3)]
    for kf in kalman_filters:
        kf.x = np.array([[positions_x[0]], [0], [0], [0], [0], [0]])
        kf.F = np.array([[1, 1, 0.5, 0, 0, 0],
                         [0, 1, 1, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 1, 1, 0.5],
                         [0, 0, 0, 0, 1, 1],
                         [0, 0, 0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0, 0, 0]])
        kf.P *= 1000  # Initial uncertainty/covariance

    # Create Animation
    num_frames = len(times)
    ani = FuncAnimation(fig, update, frames=num_frames, blit=True,
                        fargs=(positions_x, positions_y, positions_z, sc, colors, kalman_filters))

    # Make GIF
    ani.save('kalman2.gif', writer='pillow', fps=30)

    plt.show()