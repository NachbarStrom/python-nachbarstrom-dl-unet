import json
import os

from urllib import request

from PIL import Image


def download_data(json_fname: str, output_dirname: str):
    assert os.path.isfile(json_fname), f"{json_fname} doesnt exist."
    assert not os.path.isdir(output_dirname), f"{output_dirname} already exists."
    os.makedirs(output_dirname)

    with open(json_fname, "r") as file:
        data = json.load(file)

    for labeled_img in data:
        if "Masks" not in labeled_img:
            print(f"img doesnt have Masks: {labeled_img} ")
            continue

        img_id = labeled_img["ID"]

        if "Labeled Data" in labeled_img:
            res = request.urlopen(labeled_img["Labeled Data"])
            out_fname = img_id + "_original.png"
            full_output_fname = os.path.join(output_dirname, out_fname)
            Image.open(res).save(full_output_fname)

        masks = labeled_img["Masks"]

        if "Unusable Area" in masks:
            unusable_area_img_url = masks["Unusable Area"]
            response = request.urlopen(unusable_area_img_url)
            output_fname = img_id + "_unusable_area.png"
            full_output_fname = os.path.join(output_dirname, output_fname)
            Image.open(response).save(full_output_fname)

        if "Suitable Area" in masks:
            suitable_area_img_url = masks["Suitable Area"]
            response = request.urlopen(suitable_area_img_url)
            output_fname = img_id + "_suitable_area.png"
            full_output_fname = os.path.join(output_dirname, output_fname)
            Image.open(response).save(full_output_fname)

        print(f"image fetched: {img_id}")


if __name__ == '__main__':
    json_fname = "/home/tomas/Desktop/labelbox/2018-08-26_44_images_labeled.json"
    output_dirname = "/home/tomas/Desktop/labelbox-download"
    download_data(json_fname, output_dirname)
