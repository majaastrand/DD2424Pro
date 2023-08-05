import os

# dir = "C:\Users\Maja\Documents\Skola\År4\Deep\oxford-iiit-pet\images"


""" for filename in os.listdir(dir):
    os.rename(filename, filename.replace('_intsect_d', ''))






Abyssinian = []


unique_name = []
for item in filename_list:
    if item not in unique_name:
        unique_name.append(item)
dog_list = []
cat_list = []
for item in unique_name:
    if item[0].isupper():
        cat_list.append(item)
    else:
        dog_list.append(item) """

# Your code here

num = 0
filename_list = []
for root, dirs, files in os.walk("C:\\Users\\Maja\\Documents\\Skola\\År4\\Deep\\oxford-iiit-pet\\images"):
    for file in files:
        name = file
        name = name.replace("_", '')
        name = name.replace("1", '')
        name = name.replace("2", '')
        name = name.replace("3", '')
        name = name.replace("4", '')
        name = name.replace("5", '')
        name = name.replace("6", '')
        name = name.replace("7", '')
        name = name.replace("8", '')
        name = name.replace("9", '')
        name = name.replace("0", '')

        filename_list.append(name)


print(filename_list[300])
