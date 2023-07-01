import io
import os
import time
import json
import argparse
import requests

from PIL import Image
from tqdm import tqdm
from pathlib import Path

def save_image(id2image_url, args):
    for image_id, image_url in tqdm(id2image_url.items(), ncols=50):
        fname = f"{image_id}_{image_url.split('/')[-1]}"
        request_trial = 0
        while True:
            try:
                response = requests.get(image_url)
                # check if image was downloaded (response must be 200). one exception:
                # imgur gives response 200 with "removed.png" image if not found.
                if response.status_code != 200:
                    print(f'Wrong status_code = {response.status_code}')
                    if response.status_code == 429:
                        time.sleep(10)
                elif "removed.png" in response.url:
                    print(f"Removed image: {image_url}")
                    break
                else:
                    # Write image to disk if it was downloaded successfully.
                    pil_image = Image.open(io.BytesIO(response.content)).convert("RGB")
                    image_width, image_height = pil_image.size
                    scale = args.longer_resize / float(max(image_width, image_height))
                    if scale != 1.0:
                        new_width, new_height = tuple(
                            int(round(d * scale)) for d in (image_width, image_height)
                        )
                        pil_image = pil_image.resize((new_width, new_height))
                    pil_image.save(os.path.join(args.save_dialog_image_directory, fname))
                    break
            except:
                print('Something wrong...')

            request_trial += 1
            print(f'{request_trial}-th Request trial...')
            if request_trial > args.max_trial:
                break
    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default=0, type=int)
    parser.add_argument("--num_splits", default=1, type=int)
    parser.add_argument("--max_trial", default=5, type=int)
    parser.add_argument("--dialog_image_url_directory", default=None, required=True, type=str)
    parser.add_argument("--persona_image_url_directory", default=None, required=True, type=str)
    parser.add_argument("--save_dialog_image_directory", default=None, required=True, type=str)
    parser.add_argument("--save_persona_image_directory", default=None, required=True, type=str)
    parser.add_argument("--longer_resize", default=512, type=int,
                        help="Resize the longer edge of image to this size before \
                        saving to disk (preserve aspect ratio). Set to -1 to avoid any resizing. \
                        Defaults to 512.")
    args = parser.parse_args()

    with open(args.dialog_image_url_directory, 'r') as fp:
        id2dialog_image_url = json.load(fp)

    with open(args.persona_image_url_directory, 'r') as fp:
        id2persona_image_url = json.load(fp)

    save_image(id2dialog_image_url, args)
    save_image(id2persona_image_url, args)

    print('Good Job Computer!')

if __name__ == '__main__':
    main()
