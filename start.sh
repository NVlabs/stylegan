virtualenv -p /usr/bin/python3 py3env
source py3env/bin/activate
python -V
pip install scipy requests pillow lmdb opencv-python
apt update && apt install -y libsm6 libxext6
apt-get install -y libsm6 libxrender1 libfontconfig1
python train.py
