Wikipainting Dataset
==================

The step for training Wikipainting dataset is as follows:
1. run `data_provider/download_dataset.py` to download the images of the dataset.
1. run `data_provider/generate_wiki_data` to crop and pack images into lmdb.
1. run `wiki_run.sh` to train OPN, TPN, and MNet.