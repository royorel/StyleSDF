import os
import html
import glob
import uuid
import hashlib
import requests
from tqdm import tqdm
from pdb import set_trace as st


ffhq_full_model_spec = dict(file_url='https://drive.google.com/uc?id=13s_dH768zJ3IHUjySVbD1DqcMolqKlBi',
                            alt_url='', file_size=202570217, file_md5='1ab9522157e537351fcf4faed9c92abb',
                            file_path='full_models/ffhq1024x1024.pt',)
ffhq_volume_renderer_spec = dict(file_url='https://drive.google.com/uc?id=1zzB-ACuas7lSAln8pDnIqlWOEg869CCK',
                                 alt_url='', file_size=63736197, file_md5='fe62f26032ccc8f04e1101b07bcd7462',
                                 file_path='pretrained_renderer/ffhq_vol_renderer.pt',)
afhq_full_model_spec = dict(file_url='https://drive.google.com/uc?id=1jZcV5__EPS56JBllRmUfUjMHExql_eSq',
                            alt_url='', file_size=184908743, file_md5='91eeaab2da5c0d2134c04bb56ac5aeb6',
                            file_path='full_models/afhq512x512.pt',)
afhq_volume_renderer_spec = dict(file_url='https://drive.google.com/uc?id=1xhZjuJt_teghAQEoJevAxrU5_8VqqX42',
                                 alt_url='', file_size=63736197, file_md5='cb3beb6cd3c43d9119356400165e7f26',
                                 file_path='pretrained_renderer/afhq_vol_renderer.pt',)
volume_renderer_sphere_init_spec = dict(file_url='https://drive.google.com/uc?id=1CcYWbHFrJyb4u5rBFx_ww-ceYWd309tk',
                                        alt_url='', file_size=63736197, file_md5='a6be653435b0e49633e6838f18ff4df6',
                                        file_path='pretrained_renderer/sphere_init.pt',)


def download_pretrained_models():
    print('Downloading sphere initialized volume renderer')
    with requests.Session() as session:
        try:
            download_file(session, volume_renderer_sphere_init_spec)
        except:
            print('Google Drive download failed.\n' \
                  'Trying do download from alternate server')
            download_file(session, volume_renderer_sphere_init_spec, use_alt_url=True)

    print('Downloading FFHQ pretrained volume renderer')
    with requests.Session() as session:
        try:
            download_file(session, ffhq_volume_renderer_spec)
        except:
            print('Google Drive download failed.\n' \
                  'Trying do download from alternate server')
            download_file(session, ffhq_volume_renderer_spec, use_alt_url=True)

    print('Downloading FFHQ full model (1024x1024)')
    with requests.Session() as session:
        try:
            download_file(session, ffhq_full_model_spec)
        except:
            print('Google Drive download failed.\n' \
                  'Trying do download from alternate server')
            download_file(session, ffhq_full_model_spec, use_alt_url=True)

    print('Done!')

    print('Downloading AFHQ pretrained volume renderer')
    with requests.Session() as session:
        try:
            download_file(session, afhq_volume_renderer_spec)
        except:
            print('Google Drive download failed.\n' \
                  'Trying do download from alternate server')
            download_file(session, afhq_volume_renderer_spec, use_alt_url=True)

    print('Done!')

    print('Downloading Downloading AFHQ full model (512x512)')
    with requests.Session() as session:
        try:
            download_file(session, afhq_full_model_spec)
        except:
            print('Google Drive download failed.\n' \
                  'Trying do download from alternate server')
            download_file(session, afhq_full_model_spec, use_alt_url=True)

    print('Done!')

def download_file(session, file_spec, use_alt_url=False, chunk_size=128, num_attempts=10):
    file_path = file_spec['file_path']
    if use_alt_url:
        file_url = file_spec['alt_url']
    else:
        file_url = file_spec['file_url']

    file_dir = os.path.dirname(file_path)
    tmp_path = file_path + '.tmp.' + uuid.uuid4().hex
    if file_dir:
        os.makedirs(file_dir, exist_ok=True)

    progress_bar = tqdm(total=file_spec['file_size'], unit='B', unit_scale=True)
    for attempts_left in reversed(range(num_attempts)):
        data_size = 0
        progress_bar.reset()
        try:
            # Download.
            data_md5 = hashlib.md5()
            with session.get(file_url, stream=True) as res:
                res.raise_for_status()
                with open(tmp_path, 'wb') as f:
                    for chunk in res.iter_content(chunk_size=chunk_size<<10):
                        progress_bar.update(len(chunk))
                        f.write(chunk)
                        data_size += len(chunk)
                        data_md5.update(chunk)

            # Validate.
            if 'file_size' in file_spec and data_size != file_spec['file_size']:
                raise IOError('Incorrect file size', file_path)
            if 'file_md5' in file_spec and data_md5.hexdigest() != file_spec['file_md5']:
                raise IOError('Incorrect file MD5', file_path)
            break

        except:
            # Last attempt => raise error.
            if not attempts_left:
                raise

            # Handle Google Drive virus checker nag.
            if data_size > 0 and data_size < 8192:
                with open(tmp_path, 'rb') as f:
                    data = f.read()
                links = [html.unescape(link) for link in data.decode('utf-8').split('"') if 'confirm=t' in link]
                if len(links) == 1:
                    file_url = requests.compat.urljoin(file_url, links[0])
                    continue

    progress_bar.close()

    # Rename temp file to the correct name.
    os.replace(tmp_path, file_path) # atomic

    # Attempt to clean up any leftover temps.
    for filename in glob.glob(file_path + '.tmp.*'):
        try:
            os.remove(filename)
        except:
            pass

if __name__ == "__main__":
    download_pretrained_models()
