AVA Style Dataset
==================

The step for training AVA Style Dataset is as follows:
1. run `data_provider/download_ava.py` to download the images of the dataset.
1. run `data_provider/generate_ava_data.py` to crop and pack images into lmdb.
1. run `models/download.sh` to download VGG19 model.
1. run `ava_run.sh` to train OPN, TPN, and MNet.
1. run `ava_test_XXX.py` to test OPN, TPN and MNet.