import numpy
import imageio
import numpy as np
from PIL import Image
from pathlib import Path
import argparse


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--dir', type=str, default='./')
    parser.add_argument('--fps', type=float, default=20.0)
    parser.add_argument('--out', type=str, default='./')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_config()
    path = Path(args.dir)
    save = Path(args.out)
    save.mkdir(exist_ok=True, parents=True)
    print(path)
    file_it = list(path.iterdir())
    file_it.sort()
    image_list = []
    for file in file_it:
        file = file.__str__()
        print(file)
        image = Image.open(file)
        image = np.asarray(image)
        image_list.append(image)
    save = save / f'{path.stem}.gif'
    print(f'save to {save}')
    imageio.mimsave(save, image_list[::6], fps=args.fps)
