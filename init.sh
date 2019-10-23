git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ --user
cd ..
git clone https://github.com/mapillary/inplace_abn.git
cd inplace_abn
python setup.py install --user
cd ..
pip install torch-encoding==0.4.5 --user
pip install tensorboardX --user
