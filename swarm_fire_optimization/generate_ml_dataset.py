import numpy as np
import os


def build_dataset(tlbo_result_file):
    """
    Build (state -> control) dataset from TLBO results
    """
    data = np.load(tlbo_result_file, allow_pickle=True)

    X = data["X"]   # (T+1, N, 2)
    U = data["U"]   # (T, N, 2)

    T, N, _ = U.shape
    inputs, outputs = [], []

    for t in range(T):
        fire_center = np.mean(X[t], axis=0)

        for i in range(N):
            xi, yi = X[t, i]
            xf, yf = fire_center
            inputs.append([xi, yi, xf, yf, t])
            outputs.append(U[t, i])

    return np.array(inputs), np.array(outputs)


def save_dataset(output_path="ml_dataset.npz"):
    tlbo_file = "tlbo_caseB_full.npz"
    if not os.path.exists(tlbo_file):
        raise FileNotFoundError(f"'{tlbo_file}' not found.")

    X, y = build_dataset(tlbo_file)
    np.savez(output_path, X=X, y=y)

    print(f"ML dataset saved to '{output_path}'")
    print(f"Samples: {X.shape[0]}")


def load_dataset(path="ml_dataset.npz"):
    data = np.load(path)
    return data["X"], data["y"]


def main():
    save_dataset()


if __name__ == "__main__":
    main()
