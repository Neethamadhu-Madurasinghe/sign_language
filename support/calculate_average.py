import re

def calculate_averages(filename):
    metrics = {"F1 Score": [], "Precision": [], "Recall": [], "Accuracy": []}
    
    with open(filename, 'r') as file:
        for line in file:
            match = re.match(r"(F1 Score|Precision|Recall|Accuracy): ([0-9\.]+)", line.strip())
            if match:
                metric, value = match.groups()
                metrics[metric].append(float(value))
    
    averages = {key: sum(values) / len(values) for key, values in metrics.items() if values}
    
    return averages

# Example usage
filename = "metrics.txt"  # Replace with your actual filename
averages = calculate_averages(filename)

for metric, avg_value in averages.items():
    print(f"Average {metric}: {avg_value:.6f}")
