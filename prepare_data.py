import argparse
from io import BytesIO
import multiprocessing
from functools import partial

from PIL import Image
import lmdb
from tqdm import tqdm
from torchvision import datasets
from torchvision.transforms import functional as trans_fn
from pdb import set_trace as st


def resize_and_convert(img, size, resample):
    img = trans_fn.resize(img, size, resample)
    img = trans_fn.center_crop(img, size)
    buffer = BytesIO()
    img.save(buffer, format="png")
    val = buffer.getvalue()

    return val


def resize_multiple(
    img, sizes=(128, 256, 512, 1024), resample=Image.LANCZOS):
    imgs = []

    for size in sizes:
        imgs.append(resize_and_convert(img, size, resample))

    return imgs


def resize_worker(img_file, sizes, resample):
    i, file = img_file
    img = Image.open(file)
    img = img.convert("RGB")
    out = resize_multiple(img, sizes=sizes, resample=resample)

    return i, out


def prepare(
    env, dataset, n_worker, sizes=(128, 256, 512, 1024), resample=Image.LANCZOS
):
    resize_fn = partial(resize_worker, sizes=sizes, resample=resample)

    files = sorted(dataset.imgs, key=lambda x: x[0])
    files = [(i, file) for i, (file, label) in enumerate(files)]
    total = 0

    with multiprocessing.Pool(n_worker) as pool:
        for i, imgs in tqdm(pool.imap_unordered(resize_fn, files)):
            for size, img in zip(sizes, imgs):
                key = f"{size}-{str(i).zfill(5)}".encode("utf-8")

                with env.begin(write=True) as txn:
                    txn.put(key, img)

            total += 1

        with env.begin(write=True) as txn:
            txn.put("length".encode("utf-8"), str(total).encode("utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess images for model training")
    parser.add_argument("--size", type=str, default="64,512,1024", help="resolutions of images for the dataset")
    parser.add_argument("--n_worker", type=int, default=32, help="number of workers for preparing dataset")
    parser.add_argument("--resample", type=str, default="lanczos", help="resampling methods for resizing images")
    parser.add_argument("--out_path", type=str, default="datasets/FFHQ", help="Target path of the output lmdb dataset")
    parser.add_argument("in_path", type=str, help="path to the input image dataset")
    args = parser.parse_args()

    resample_map = {"lanczos": Image.LANCZOS, "bilinear": Image.BILINEAR}
    resample = resample_map[args.resample]

    sizes = [int(s.strip()) for s in args.size.split(",")]

    print(f"Make dataset of image sizes:", ", ".join(str(s) for s in sizes))

    imgset = datasets.ImageFolder(args.in_path)

    with lmdb.open(args.out_path, map_size=1024 ** 4, readahead=False) as env:
        prepare(env, imgset, args.n_worker, sizes=sizes, resample=resample)
