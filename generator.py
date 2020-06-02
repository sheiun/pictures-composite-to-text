from argparse import ArgumentParser
from glob import glob
from os import makedirs
from random import choice
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont


def to_array(text, scale, font_size) -> np.array:
    font = ImageFont.truetype("fonts/NotoSansTC-Bold.otf", font_size)
    image = Image.fromarray(np.ones((100, 100 * len(text))) * 255)
    image = image.convert("L")
    draw = ImageDraw.Draw(image)
    size = draw.textsize(text, font=font)
    draw.text((0, 0), text, font=font, fill="black")
    image = image.crop((0, 0, *size))
    image = image.resize(map(lambda x: x * scale, image.size))
    return np.asarray(image)


def brightness(image: Image, factor=1):
    """
    0 <-> 1 <-> 5
    Dark <-> Origin <-> Bright
    """
    # arr = np.asarray(image).astype(np.int32)
    # arr = (arr * 254 / 255 + 1).astype(np.uint8)
    # image = Image.fromarray(arr)
    enhancer = ImageEnhance.Brightness(image)
    output = enhancer.enhance(factor)
    return np.asarray(output)


def get_image(size: tuple, pics_path) -> Image:
    paths = glob(pics_path)
    path = choice(paths)
    img = Image.open(path)
    return img.resize(size)


def to_factor(rate: float):
    return rate / 255 * 3.8 + 0.2


def generate(text, scale=1, size=(128, 128), font_size=32, pics_path="pics/*.jpg"):
    matrix = to_array(text, scale=scale, font_size=font_size)

    uuid = uuid4()
    Image.fromarray(matrix).save(
        f"images/origin/{text}_{scale}_{font_size}_{str(uuid)}.png"
    )

    output = []
    for idx, x in enumerate(range(matrix.shape[1])):
        col = []
        for y in range(matrix.shape[0]):
            rate = matrix[y, x]
            im = brightness(get_image(size, pics_path), to_factor(rate))
            col.append(im)
        col = np.vstack(col)
        output.append(col)
        print(f"Progress: {idx+1} / {matrix.shape[1]}")
    output = np.hstack(output)

    Image.fromarray(output).save(
        f"images/transform/{text}_{size}_{str(uuid)}.png", format="png", dpi=(500, 500)
    )

if __name__ == "__main__":
    try:
        makedirs("images/origin/")
        makedirs("images/transform/")
    except:
        pass
    parser = ArgumentParser()
    parser.add_argument("text", help="Text to generate.", type=str)
    parser.add_argument(
        "--size",
        help="Pics load size.",
        default=(128, int(128 * 1.4)),
        nargs="+",
        type=int,
    )
    parser.add_argument("--scale", help="Scale for text.", default=1, type=float)
    parser.add_argument(
        "--font_size", help="Font size for generated text.", default=32, type=int
    )
    args = parser.parse_args()
    args = vars(args)
    generate(args.pop("text"), **args)
