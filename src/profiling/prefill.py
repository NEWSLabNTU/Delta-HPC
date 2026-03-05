import json
import argparse
import numpy as np
import matplotlib
# Use 'Agg' backend to save files without a display/GUI
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Extract vLLM prefill parameters alpha and beta.")
    parser.add_argument("--input", type=str, required=True, help="Path to the benchmark_detailed_results.json file")
    parser.add_argument("--output-param", type=str, required=True, help="Output path for parameter JSON")
    parser.add_argument("--output-plot", type=str, required=True, help="Output path for the fit plot")
    args = parser.parse_args()

    # 1. Load and Parse the benchmark file
    with open(args.input, 'r') as f:
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
        r_squared = correlation_matrix[0, 1]**2

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
            "unit_beta": "seconds_per_token"
        }
        with open(args.output_param, 'w') as f:
            json.dump(output_data, f, indent=4)
        print(f"Parameters saved to: {args.output_param}")

        # 4. Generate and Save Plot
        plt.figure(figsize=(10, 6))
        plt.scatter(X, Y, alpha=0.6, label='Benchmark Data', color='blue')
        
        # Plot regression line
        line_x = np.array([min(X), max(X)])
        line_y = beta * line_x + alpha
        plt.plot(line_x, line_y, color='red', label=f'Fit: y = {beta:.2e}x + {alpha:.4f}')

        plt.xlabel('Prompt Tokens (x)')
        plt.ylabel('Prefill Time (y) [seconds]')
        plt.title('vLLM Prefill Time Profiling: $y = \\alpha + \\beta x$')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.savefig(args.output_plot)
        print(f"Fit figure saved to: {args.output_plot}")
        plt.close()

    else:
        print("Insufficient data points for regression. Need at least 2 successful requests.")

if __name__ == "__main__":
    main()
