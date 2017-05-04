mkdir -p images/
wget http://phototriage.cs.princeton.edu/data/train_val.zip && unzip train_val.zip
mv train_val/train_val_imgs/* images/
mv train_val/train_pairlist.txt train.txt
mv train_val/val_pairlist.txt valid.txt
rm -rf train_val.zip train_val/
wget http://phototriage.cs.princeton.edu/data/test.zip && unzip test.zip
mv test/test_imgs/* images/
mv test/test_pairlist.txt test.txt
rm -rf test.zip test/