Wikipainting Dataset
==================

The step for training Wikipainting Dataset is as follows:
1. run `data_provider/download_dataset.py` to download the images of the dataset.
1. run `data_provider/generate_wiki_data.py` to crop and pack images into lmdb.
1. run `models/download.sh` to download VGG19 model.
1. run `wiki_run.sh` to train OPN, TPN, and MNet.