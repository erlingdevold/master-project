# #!/bin/bash

python3 processing/collator.py

echo "Running training for all models"

python3 processing/train.py --head_type binary --onehot 1 --threshold 1 --temporal 0
python3 processing/train.py --head_type binary --onehot 1 --threshold 5 --temporal 0
python3 processing/train.py --head_type binary --onehot 1 --threshold 10 --temporal 0
python3 processing/train.py --head_type binary --onehot 1 --threshold 20 --temporal 0

python3 processing/train.py --head_type binary --onehot 1 --threshold 1 --temporal 1
python3 processing/train.py --head_type binary --onehot 1 --threshold 5 --temporal 1
python3 processing/train.py --head_type binary --onehot 1 --threshold 10 --temporal 1
python3 processing/train.py --head_type binary --onehot 1 --threshold 20 --temporal 1

python3 processing/train.py --head_type multi --onehot 0 --threshold 1 --temporal 0
python3 processing/train.py --head_type multi --onehot 0 --threshold 5 --temporal 0
python3 processing/train.py --head_type multi --onehot 0 --threshold 10 --temporal 0
python3 processing/train.py --head_type multi --onehot 0 --threshold 20 --temporal 0

python3 processing/train.py --head_type multi --onehot 1 --threshold 1 --temporal 0
python3 processing/train.py --head_type multi --onehot 1 --threshold 5 --temporal 0
python3 processing/train.py --head_type multi --onehot 1 --threshold 10 --temporal 0
python3 processing/train.py --head_type multi --onehot 1 --threshold 20 --temporal 0

python3 processing/train.py --head_type multi --onehot 1 --threshold 1 --temporal 1
python3 processing/train.py --head_type multi --onehot 1 --threshold 5 --temporal 1
python3 processing/train.py --head_type multi --onehot 1 --threshold 10 --temporal 1
python3 processing/train.py --head_type multi --onehot 1 --threshold 20 --temporal 1

python3 processing/train.py --head_type multi --onehot 0 --threshold 1 --temporal 1
python3 processing/train.py --head_type multi --onehot 0 --threshold 5 --temporal 1
python3 processing/train.py --head_type multi --onehot 0 --threshold 10 --temporal 1
python3 processing/train.py --head_type multi --onehot 0 --threshold 20 --temporal 1