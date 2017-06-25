./flickr_object_run.sh

python flickr_feature_generator.py
./flickr_mlp_run.sh
python flickr_surgury_texture.py
./flickr_texture_run.sh

python flickr_surgury_full.py
./flickr_full_run.sh