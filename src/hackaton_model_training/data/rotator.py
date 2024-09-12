from pycocotools.coco import COCO
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

input_path = "data/raw/cleared_data"
output_path = "data/processed"

def distance(p1, p2):
    return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

data_folders = ['remain', 'replace']
image_counters = {k: 1 for k in ['replace_side', 'replace_top', 'remain_side', 'remain_top']}
distributions = {k: [] for k in ['side_height', 'side_width', 'bottom_height', 'bottom_width']}

for folder in data_folders:

    suffixes = set([filename[-1] for filename in os.listdir(f'{input_path}/{folder}')])

    for suffix in suffixes:

        coco_annotation_file = f'{input_path}/{folder}/annotations{suffix}/instances.json'
        coco = COCO(coco_annotation_file)

        classes_ids = coco.getCatIds()
        classes_names = coco.loadCats(classes_ids)
        id_to_name = {elem['id'] : elem['supercategory'] for elem in classes_names}
        annotation_ids = coco.getAnnIds()
        annotations = coco.loadAnns(annotation_ids)

        for annotation in tqdm(annotations):

            keypoints = annotation['segmentation'][0]
            # пропускаем не четырехугольники
            if len(keypoints) != 8:
                continue

            bbox = annotation['bbox']
            img_id = annotation['image_id']
            img_class = id_to_name[annotation['category_id']]
            img_info = coco.imgs[img_id]
            file_name = img_info['file_name']
            img = cv2.imread(f'{input_path}/{folder}/images{suffix}/{file_name}')

            inpts = np.float32([[keypoints[i], keypoints[i+1]] for i in range(0, len(keypoints), 2)])
            length1 = distance(inpts[0], inpts[1]) 
            length2 = distance(inpts[2], inpts[3])
            mean_width = (length1 + length2) / 2

            length3 = distance(inpts[1], inpts[2])
            length4 = distance(inpts[3], inpts[0])
            mean_height = (length3 + length4) / 2
            
            outpts = np.array([
                [0, 0],
                [mean_width, 0],
                [mean_width, mean_height],
                [0, mean_height]
            ], dtype='float32')

            M = cv2.getPerspectiveTransform(inpts, outpts)
            warpimg = cv2.warpPerspective(img, M, (int(mean_width), int(mean_height)))

            if mean_height > mean_width:
                warpimg = cv2.rotate(warpimg, cv2.ROTATE_90_CLOCKWISE)

            class_folder = 'bottom' if img_class == 'top' else img_class
            file_name = image_counters[f'{folder}_{img_class}']
            save_path = f"{output_path}/{class_folder}/{folder}"
            if not os.path.exists(save_path):	
                os.makedirs(save_path)	
            cv2.imwrite(f'{save_path}/{file_name}.jpg', warpimg)
            image_counters[f'{folder}_{img_class}'] += 1

            #распределение по bbox получившихся кусков
            distributions[f'{class_folder}_height'].append(bbox[3])
            distributions[f'{class_folder}_width'].append(bbox[2])


for k in distributions.keys():
    plt.hist(distributions[k], bins=50)
    x_label = 'Высота' if 'height' in k else 'Ширина'
    plt.xlabel(x_label)
    title = k.split('_')[0]
    plt.title(title)
    plt.savefig(f'{k}.png')
    plt.clf()



