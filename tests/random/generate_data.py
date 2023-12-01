import numpy as np
import random
import argparse

"""
Example Usage:
```py
python generate_data.py -n 100 -f 50 -s 0.1 -o data.txt
```
or if you have enough compute, then you can generate larger tests:
```py
python3 generate_data.py -n 10000 -f 500 -s 0.1 -o data.txt
```
"""

np.random.seed(0)

def generate_data(num_samples, num_features, sparsity, output_file):
    X = np.zeros((num_samples, num_features))
    y = np.random.randint(-10, 11, num_samples)

    for i in range(num_samples):
        for j in range(num_features):
            if random.random() < sparsity:
                X[i, j] = np.random.uniform(-10, 10)

    with open(output_file, 'w') as f:
        for i in range(num_samples):
            f.write(f'{y[i]}')
            for j in range(num_features):
                if X[i, j] != 0:
                    f.write(f' {j}:{X[i, j]:.5f}')
            f.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate test data.')
    parser.add_argument('-n', '--num_samples', type=int, required=True, help='Number of samples')
    parser.add_argument('-f', '--num_features', type=int, required=True, help='Number of features')
    parser.add_argument('-s', '--sparsity', type=float, required=True, help='Sparsity of the data (0 to 1)')
    parser.add_argument('-o', '--output_file', type=str, required=True, help='Output file in libSVM format')

    args = parser.parse_args()
    generate_data(args.num_samples, args.num_features, args.sparsity, args.output_file)
