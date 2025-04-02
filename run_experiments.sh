#!/bin/bash

script1="gnn_experiment.py"
script2="contconv_experiment.py"

# Execute the first script
echo "Executing $script1..."
python "$script1"

# Execute the second script
echo "Executing $script2..."
python "$script2"

echo "Both scripts executed successfully."
