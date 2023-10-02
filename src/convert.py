# https://www.kaggle.com/datasets/bsridevi/modes-dataset-of-stray-animals

import os
import shutil
from urllib.parse import unquote, urlparse

import numpy as np
import supervisely as sly
from cv2 import connectedComponents
from dataset_tools.convert import unpack_if_archive
from dotenv import load_dotenv
from supervisely.io.fs import (
    dir_exists,
    file_exists,
    get_file_ext,
    get_file_name,
    get_file_name_with_ext,
    get_file_size,
)
from tqdm import tqdm

import src.settings as s


def count_files(path, extension):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    # project_name = "MoDES Dataset of Cattle"
    dataset_path = "/mnt/d/datasetninja-raw/modes-cattle/out2"
    batch_size = 30
    ds_name = "ds"
    images_folder = "images"
    depths_folder = "depth"
    masks_folder = "masks"
    image_prefix = "fgbg"
    mask_prefix = "mask"
    group_tag_name = "image_id"

    def fix_masks(image_np: np.ndarray) -> np.ndarray:
        lower_bound = np.array([182, 182, 182])
        upper_bound = np.array([255, 255, 255])
        condition_white = np.logical_and(
            np.all(image_np >= lower_bound, axis=2), np.all(image_np <= upper_bound, axis=2)
        )

        lower_bound = np.array([0, 0, 0])
        upper_bound = np.array([64, 64, 64])
        condition_black = np.logical_and(
            np.all(image_np >= lower_bound, axis=2), np.all(image_np <= upper_bound, axis=2)
        )

        image_np[np.where(condition_white)] = (255, 255, 255)
        image_np[np.where(condition_black)] = (0, 0, 0)

        return image_np

    def create_ann(image_path):
        labels = []
        tags = []

        id_data = get_file_name(image_path)
        group_id = sly.Tag(tag_id, value=id_data)
        tags.append(group_id)

        image_name = get_file_name_with_ext(image_path)

        mask_path = os.path.join(masks_path, mask_prefix + image_name[4:])
        if file_exists(mask_path):
            mask_np = sly.imaging.image.read(mask_path)[:, :, :]
            mask_np = fix_masks(mask_np)[:, :, 0]
            img_height = mask_np.shape[0]
            img_wight = mask_np.shape[1]
            mask = mask_np == 255
            curr_bitmap = sly.Bitmap(mask)
            curr_label = sly.Label(curr_bitmap, obj_class)
            labels.append(curr_label)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels, img_tags=tags)

    obj_class = sly.ObjClass("cattle", sly.Bitmap)
    tag_id = sly.TagMeta("image_id", sly.TagValueType.ANY_STRING)

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
    meta = sly.ProjectMeta(obj_classes=[obj_class], tag_metas=[tag_id])
    api.project.update_meta(project.id, meta.to_json())
    api.project.images_grouping(id=project.id, enable=True, tag_name=group_tag_name)

    images_path = os.path.join(dataset_path, images_folder)
    depths_path = os.path.join(dataset_path, depths_folder)
    masks_path = os.path.join(dataset_path, masks_folder)

    dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)

    for curr_images_path in [images_path, depths_path]:
        images_names = [im_name for im_name in os.listdir(curr_images_path)]

        progress = sly.Progress("Create dataset {}".format(ds_name), len(images_names))

        for images_names_batch in sly.batched(images_names, batch_size=batch_size):
            img_pathes_batch = [
                os.path.join(curr_images_path, image_name) for image_name in images_names_batch
            ]

            images_names_batch = [
                curr_images_path.split("/")[-1] + "_" + get_file_name_with_ext(im_path)
                for im_path in img_pathes_batch
            ]

            img_infos = api.image.upload_paths(dataset.id, images_names_batch, img_pathes_batch)
            img_ids = [im_info.id for im_info in img_infos]

            anns = [create_ann(image_path) for image_path in img_pathes_batch]
            api.annotation.upload_anns(img_ids, anns)

            progress.iters_done_report(len(images_names_batch))
    return project
