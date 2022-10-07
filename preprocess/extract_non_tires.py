import os
import random
from PIL import Image

BASE_WIDTH = 800
BASE_HEIGHT = 600


def crop(imagepath):
    image = Image.open(imagepath)
    left = random.randint(250, BASE_WIDTH - 100)
    top = random.randint(408, BASE_HEIGHT - 100)
    right = left + random.randint(64, 100)
    bottom = top + random.randint(64, 100)
    im1 = image.crop((left, top, right, bottom))
    im1.save('nollantas/' + imagepath)


def analyze_file(filename):
    with open(filename) as file:
        crop(filename)


def main(foldername):
    for filename in os.listdir(foldername):
        if not filename.endswith('.jpg'):
            continue

        # Si el archivo está marcado, salteselo
        txt_file = filename.replace('.jpg', '.txt')
        if os.path.getsize(txt_file) > 0:
            continue
        print(filename)

        if os.path.getsize(filename) == 0:
            continue

        if not filename.startswith('2022'):
            continue

        analyze_file(filename)


if __name__ == '__main__':
    if not os.path.isdir('nollantas'):
        os.mkdir('nollantas')
    main('.')
