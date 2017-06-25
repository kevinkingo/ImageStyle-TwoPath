Flickr Style Dataset
==================

The step for training Flickr Style Dataset is as follows:
1. run `data_provider/assemble_data.py` to download the images of the dataset.
1. run `data_provider/generate_flickr_data.py` to crop and pack images into lmdb.
1. run `models/download.sh` to download VGG19 model.
1. run `flickr_run.sh` to train OPN, TPN, and MNet.