import numpy as np


def smile(N, bandwidth = 2.0, **kwargs):
    small = int(np.ceil(N ** (1.0/2)))
    eye_points = kwargs["eye_points"] if ("eye_points" in kwargs) else small
    mouth_points = kwargs["mouth_points"] if ("mouth_points" in kwargs) else int(np.ceil(N/10.0))
    face_points = N - 2 * eye_points - mouth_points

    X = np.zeros((N, 2))
    idx = 0

    # Eyes
    for x_shift in [-4.0, 4.0]:
        for i in range(eye_points):
            while True:
                x = 2 * np.random.rand() - 1
                y = 2 * np.random.rand() - 1
                if x**2 + y**2 <= 1.0:
                    X[idx, 0] = x + x_shift
                    X[idx, 1] = y + 4.0
                    idx += 1
                    break

    # Mouth
    for x in list(np.linspace(-5.0, 5.0, mouth_points)):
        X[idx, 0] = x
        X[idx, 1] = x**2 / 16.0 - 5.0
        idx += 1

    # Face
    for theta in list(np.linspace(0, 2*np.pi, face_points)):
        X[idx, 0] = 10.0 * np.cos(theta)
        X[idx, 1] = 10.0 * np.sin(theta)
        idx += 1

    return X, bandwidth

def robspiral(N):
    times = np.linspace(0, 2, N)
    times = times ** 6
    times = times[::-1]
    x = np.exp(.2 * times) * np.cos(times)
    y = np.exp(.2 * times) * np.sin(times)
    X = np.column_stack((x,y))
    bandwidth = 1000
    return X, bandwidth


def outliers(N, num_outliers = 50):
    X = 0.5*np.random.randn(N, 20)/np.sqrt(20.0)
    X[np.random.choice(range(N), size = num_outliers, replace = False),:] += 100.0 * np.random.randn(num_outliers, 20)
    return X, None