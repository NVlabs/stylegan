from dnnlib.util import open_url

result = (open_url("https://drive.google.com/open?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ", cache_dir="cache"))

print("data type of result:",type(result))

print("\n\nThis is the result:",result)
