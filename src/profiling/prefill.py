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
        description="Extract vLLM prefill parameters alpha and beta."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the benchmark_detailed_results.json file",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Output directory"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    if not input_path.exists():
        print(f"Error: {input_path} does not exist.")
        return -1
    if input_path.suffix != ".json":
        print(f"Error: {input_path} suffix is not json.")
        return -1
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and Parse the benchmark file
    with open(input_path, "r") as f:
        data = json.load(f)

    x_tokens = data["input_lens"]
    y_latency = data["ttfts"]

    X = np.array(x_tokens)
    Y = np.array(y_latency)

    if len(X) > 1:
        # 2. Perform Linear Regression: y = beta*x + alpha
        beta, alpha = np.polyfit(X, Y, 1)

        # Calculate R-squared for quality check
        correlation_matrix = np.corrcoef(X, Y)
        r_squared = correlation_matrix[0, 1] ** 2

        print("-" * 40)
        print("PREFILL PROFILE RESULTS")
        print("-" * 40)
        print(f"Alpha (Fixed Overhead): {alpha:.6f} s")
        print(f"Beta (Cost per Token):  {beta:.6f} s/token")
        print(f"R-squared:              {r_squared:.4f}")
        print("-" * 40)

        # 3. Save Parameters to JSON
        output_data = {
            "alpha": alpha,
            "beta": beta,
            "r_squared": r_squared,
            "unit_alpha": "seconds",
            "unit_beta": "seconds_per_token",
        }
        param_save_path = output_dir / (input_path.stem + "-param.json")
        with open(param_save_path, "w") as f:
            json.dump(output_data, f, indent=4)
        print(f"Parameters saved to: {param_save_path}")

        # 4. Generate and Save Plot
        plt.figure(figsize=(10, 6))
        plt.scatter(X, Y, alpha=0.6, label="Benchmark Data", color="blue")

        # Plot regression line
        line_x = np.array([min(X), max(X)])
        line_y = beta * line_x + alpha
        plt.plot(
            line_x, line_y, color="red", label=f"Fit: y = {beta:.2e}x + {alpha:.4f}"
        )

        plt.xlabel("Prompt Tokens (x)")
        plt.ylabel("Prefill Time (y) [seconds]")
        plt.title("vLLM Prefill Time Profiling: $y = \\alpha + \\beta x$")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)

        plot_save_path = output_dir / (input_path.stem + "-plot.png")
        plt.savefig(plot_save_path)
        print(f"Fit figure saved to: {plot_save_path}")
        plt.close()

    else:
        print(
            "Insufficient data points for regression. Need at least 2 successful requests."
        )


if __name__ == "__main__":
    main()
