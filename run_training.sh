# #!/bin/bash

echo "Running training for all models"

python3 processing/train.py --head_type binary --onehot 1 --threshold 1 --temporal 0
