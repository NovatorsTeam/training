from pycocotools.coco import COCO
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

data_folders = ['remain', 'replace']
image_counters = {k: 1 for k in ['replace_side', 'replace_top', 'remain_side', 'remain_top']}
distributions = {k: [] for k in ['side_height', 'side_width', 'bottom_height', 'bottom_width']}

for folder in data_folders:

    suffixes = set([filename[-1] for filename in os.listdir(f'./data/{folder}')])

    for suffix in suffixes:

        coco_annotation_file = f'./data/{folder}/annotations{suffix}/instances.json'
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

            inpts = np.float32([[keypoints[i], keypoints[i+1]] for i in range(0, len(keypoints), 2)])
            ignored_outpts = np.float32([[0,0],[0,320],[320,320],[320,0]])
            
            # размеры по bbox учитываются только при распределении
            # обрезка стоит 320 на 320
            # при необходимости сделать название переменной выше ignored, а ниже просто outpts
            outpts = np.float32([
                [bbox[0], bbox[1]], 
                [bbox[0], bbox[1] + bbox[3]], 
                [bbox[0] + bbox[2], bbox[1] + bbox[3]],
                [bbox[0] + bbox[2], bbox[1]]
            ])

            M = cv2.getPerspectiveTransform(inpts, outpts)
            file_name = img_info['file_name']
            img = cv2.imread(f'./data/{folder}/images{suffix}/{file_name}')

            img_width = int(bbox[2])
            img_height = int(bbox[3])
            warpimg = cv2.warpPerspective(img, M, (img_width, img_height))

            class_folder = 'bottom' if img_class == 'top' else img_class
            file_name = image_counters[f'{folder}_{img_class}']
            cv2.imwrite(f'./outputs/{class_folder}/{folder}/{file_name}.jpg', warpimg)
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



