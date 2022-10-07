import os
from PIL import Image

BASE_WIDTH = 800
BASE_HEIGHT = 600


def resize(imagepath, x_pos, y_pos, width, height, ocurrencia):
    image = Image.open(imagepath)
    left = int(x_pos * BASE_WIDTH - width / 2)
    top = int(y_pos * BASE_HEIGHT - height / 2)
    right = int(x_pos * BASE_WIDTH + width / 2)
    bottom = int(y_pos * BASE_HEIGHT + height / 2)
    im1 = image.crop((left, top, right, bottom))
    im1.save('llantas/' + imagepath.replace('jpg', f'{ocurrencia}.jpg'))


def analyze_file(filename):
    with open(filename) as file:
        ocurrencia = 0
        for line in file:
            data = [float(dato) for dato in line.split(' ')]
            width = data[3] * BASE_WIDTH
            if width < 64:
                continue

            height = data[4] * BASE_HEIGHT
            if height < 64:
                continue

            imagepath = filename.replace('.txt', '.jpg')
            resize(imagepath, data[1], data[2], width, height, ocurrencia)
            ocurrencia += 1


def main(foldername):
    for filename in os.listdir(foldername):
        if not filename.endswith('.txt'):
            continue

        if os.path.getsize(filename) == 0:
            continue

        if not filename.startswith('2022'):
            continue

        analyze_file(filename)


if __name__ == '__main__':
    if not os.path.isdir('llantas'):
        os.mkdir('llantas')
    main('.')
