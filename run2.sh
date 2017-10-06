rm -rf tf_logs/ models
mkdir models

cp pngfilenames_generating_train.py data/train/.
cp pngfilenames_generating_valid.py data/valid/.
cp pngfilenames_generating_test.py data/test/.


cd data/train/
python pngfilenames_generating_train.py
cd ..
cd valid
python pngfilenames_generating_valid.py
cd ..
cd test
python pngfilenames_generating_test.py
cd ..
cd ..
mv data/train/train.txt .
mv data/valid/valid.txt .
mv data/test/test.txt .

export CUDA_VISIBLE_DEVICES=1

python trainer2.py
