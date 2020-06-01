from dnnlib.util import open_url
import pickle

result = (open_url("https://drive.google.com/open?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ", cache_dir="cache"))

print("data type of result:",type(result))

print("\n\nThis is the result:",result)

with open("/drive/My Drive/Colab_folder/stylegan/stylegan_pretrained_nividia.pkl") as f:
  datums = pickle.load(f)

print("type of data:",type(datums))
print("\n\ncontent of data:",datums)
