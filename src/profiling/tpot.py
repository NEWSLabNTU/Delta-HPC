import re
import json
from pathlib import Path
import argparse
import numpy as np
import matplotlib

# Use 'Agg' backend to save files without a display/GUI
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Extract vLLM TPOT parameters alpha, beta, and gaussian noise sigma from sweep results."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing the benchmark result JSON files",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Output directory"
    )
    args = parser.parse_args()

    input_dir_path = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir_path.is_dir():
        print(f"Error: {input_dir_path} is not a directory.")
        return -1
    output_dir.mkdir(parents=True, exist_ok=True)

    # We will store the mean ITL of each request matched with its concurrency
    all_concurrency = []
    all_request_mean_itls = []

    # 1. Check directory and find JSON files
    json_files = list(input_dir_path.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in directory: {input_dir_path}")
        return
    print(f"Found {len(json_files)} result files in {input_dir_path}.")

    for file_path in json_files:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            # 1. Get concurrency from the 'max_concurrency' key
            n_val = data.get("max_concurrency")

            # 2. Get the list of lists of ITLs (one list per request)
            nested_itls = data.get("itls", [])

            if n_val is not None and nested_itls:
                for request_itls in nested_itls:
                    if request_itls:  # Ensure the list isn't empty
                        req_mean_s = sum(request_itls) / len(request_itls)

                        all_concurrency.append(n_val)
                        all_request_mean_itls.append(req_mean_s)
            else:
                print(
                    f"Warning: Skipping {file_path} - Missing 'max_concurrency' or 'itls'."
                )

        except Exception as e:
            print(f"Skipping {file_path} due to error: {e}")

    if not all_concurrency:
        print(
            "No valid data points extracted. Ensure JSON files contain 'max_concurrency' and nested 'itls' fields."
        )
        return

    X = np.array(all_concurrency)
    Y = np.array(all_request_mean_itls)

    # 2. Perform Linear Regression: y = alpha + beta * N
    # alpha = intercept (memory wall), beta = slope (scaling cost)
    beta, alpha = np.polyfit(X, Y, 1)

    # 3. Calculate Noise (Sigma)
    # Predicted Y values based on the linear model
    Y_pred = beta * X + alpha

    # Residuals represent the 'noise' in the real data
    residuals = Y - Y_pred

    # Sigma is the standard deviation of the residuals
    sigma = np.std(residuals)

    # Calculate R-squared
    correlation_matrix = np.corrcoef(X, Y)
    r_squared = correlation_matrix[0, 1] ** 2

    print("-" * 45)
    print("TPOT (DECODING) PROFILE RESULTS (Linear + Noise)")
    print("-" * 45)
    print(f"Total Request Samples:           {len(Y)}")
    print(f"Intercept (alpha - Memory Wall): {alpha:.6f} s")
    print(f"Slope (beta - Scaling Cost):    {beta:.6e} s/request")
    print(f"Sigma (Noise StdDev):           {sigma:.6f} s")
    print(f"R-squared:                       {r_squared:.4f}")
    print("-" * 45)

    # 4. Save Parameters to JSON
    output_data = {
        "alpha": alpha,
        "beta": beta,
        "sigma": sigma,
        "r_squared": r_squared,
        "sample_size_requests": len(Y),
        "unit_alpha": "seconds",
        "unit_beta": "seconds_per_request",
        "model_formula": "ITL = alpha + beta * N + N(0, sigma^2)",
    }

    # Clean up filename for output
    if json_files:
        no_concur_filename = re.sub(r"concurrency-\d+", "", json_files[0].stem)
        param_save_path = output_dir / (no_concur_filename + "-param.json")
        with open(param_save_path, "w") as f:
            json.dump(output_data, f, indent=4)
        print(f"Parameters saved to: {param_save_path}")

    # 5. Generate and Save Plot
    plt.figure(figsize=(12, 7))

    # Scatter the request-level means (converted to ms for plot readability)
    plt.scatter(
        X, Y * 1000, alpha=0.15, label="Request Mean ITL Samples", color="blue", s=15
    )

    # Calculate global means per N for trend clarity
    unique_n = np.sort(np.unique(X))
    global_means_y = [np.mean(Y[X == n]) * 1000 for n in unique_n]
    plt.scatter(
        unique_n,
        global_means_y,
        color="black",
        marker="x",
        label="Global Mean per N",
        s=60,
        zorder=3,
    )

    # Plot regression line (converted to ms)
    line_x = np.linspace(min(X), max(X), 100)
    line_y_s = beta * line_x + alpha
    line_y_ms = line_y_s * 1000

    plt.plot(
        line_x,
        line_y_ms,
        color="red",
        linewidth=2,
        label=f"Fit: $y = {beta*1000:.4f}N + {alpha*1000:.2f}$ms",
        zorder=4,
    )

    # Visualize Noise Range (1 Sigma) in ms
    sigma_ms = sigma * 1000
    plt.fill_between(
        line_x,
        line_y_ms - sigma_ms,
        line_y_ms + sigma_ms,
        color="red",
        alpha=0.15,
        label=rf"Noise Margin ($\pm 1\sigma$: {sigma_ms:.2f}ms)",
    )

    plt.xlabel("Concurrent Requests (N)")
    plt.ylabel("Mean Inter-Token Latency per Request (ms)")
    plt.title(
        "vLLM Decoding Profiling: $y = \\alpha + \\beta N + \\epsilon$ (Request-Level Analysis)"
    )
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    plot_save_path = output_dir / (no_concur_filename + "-plot.png")
    plt.savefig(plot_save_path, dpi=150)
    print(f"Fit figure saved to: {plot_save_path}")
    plt.close()


if __name__ == "__main__":
    main()
