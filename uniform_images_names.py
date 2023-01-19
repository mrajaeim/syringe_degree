import os

for file in os.listdir('./dataset/'):
    path = f"./dataset/{file}"
    image_number, ext = file.split(".")
    deg = (int(image_number) * 360) / 250
    new_path = f"./dataset/{image_number}_v{0}_deg_{deg:0.2f}.{ext}"
    # print(new_path)
    os.rename(path, new_path)