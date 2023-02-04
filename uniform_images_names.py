import os

for file in os.listdir('./dataset/'):
    path = f"./dataset/{file}"
    image_number, version = file[0:4], file[5:7]
    deg = (int(image_number) * 360) / 250
    new_path = f"./dataset/{image_number:04}_{version}_deg_{deg:06.2f}.jpg"
    # print(new_path)
    os.rename(path, new_path)