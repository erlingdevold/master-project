# 

## Requirements
- Python 3.11.2
- PyTorch 2
- PyTorch Lightning

Preprocessing uses 3.8.9 because of PyEcholabs dependency.
Refer to the github repository for installation guide.

CUDA is required

To run experiments, run the following command:
```
$ ./run_training.sh
```

Dataset is not included in this repository. Refer to the paper for details on how to obtain the dataset.

labels/ folder collects labels from database into a labelled multidimensional xarray DataArray.

lightning_logs folder contains the logs from the training process.
metrics from all three runs are stored in the same folder.

script in lightning_logs/plots.py plots validation loss and accuracy for all three runs. as well as MSE and MAE for regression.

ds/ folder incldues labels found from the dataset. segmented .npy files are not included.
