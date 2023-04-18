from pycocotools.coco import COCO
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import albumentations
import albumentations.pytorch

class COCO_Dataset(Dataset):
        def __init__(self, root_dir='/data', set_name='train2014', split='TRAIN', transform=None):
                super().__init__()
                self.root_dir = root_dir
                self.set_name = set_name
                self.coco = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
                self.whole_image_ids = self.coco.getImgIds()
                self.transform = transform
                
        def __len__(self):
                return len(self.whole_image_ids)
        
        def load_image(self, img_idx):
                img_info = self.coco.loadImgs(self.whole_image_ids[img_idx])[0]
                image = Image.open(os.path.join(self.root_dir, 'images', img_info['file_name'])).convert('RGB')
                return image, img_info['width'], img_info['height']
        
        def load_annotation(self, img_idx, width, height):
                anns_info = self.coco.getAnnIds(self.whole_image_ids[img_idx])
                
                if len(anns_info)==0:
                        return torch.zeros((1, 5))
                annotations = []
                
                label_info = self.coco.loadAnns(anns_info)
                segmentation = np.zeros((height, width))
                for info in label_info:
                        if info['bbox'][2]<1 or info['bbox'][3]<1:
                                continue
                        annotation = np.zeros((1, 5))
                        annotation[0, :4] = info['bbox']
                        annotation[0, 4] = info['category_id']
                        annotations.extend(annotation)
                        segmentation = np.maximum(segmentation, self.coco.annToMask(info)*info['category_id'])
                annotations = np.array(annotations)
                return annotations[:, :4], annotations[:, 4], segmentation
        
        def __getitem__(self, idx):
                image, width, height = self.load_image(idx)
                bbox, class, seg = self.load_annotation(idx, width=width, height=height)
                if self.transform:
                        result = self.transform(image=image, bboxes=bbox, mask=seg)
                        return result['image'], result['bboxes'], class, result['mask']
                else:
                        return image, bbox, class, seg

def collate_fn(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    img, bbox, clas, seg = zip(*data)
    lengths = [len(cap) for cap in bbox]
    
    img = torch.stack(img, 0)
    seg = torch.stack(seg, 0)
    targets_bbox = torch.zeros(len(bbox), max(lengths), 4).long()
    for idx, cap in enumerate(bbox):
        end = lengths[idx]
        targets_bbox[idx, :end, ...] = cap[:end, ...]
        
    targets_cls = torch.zeros(len(clas), max(lengths)).long()
    for idx, cap in enumerate(clas):
        end = lengths[idx]
        targets_cls[idx, :end] = cap[:end]
    
    return img, targets_bbox, targets_cls, seg

if __name__ == '__main__':
        transform = albumentations.Compose(
            [
                albumentations.Resize(420, 420, interpolation=cv2.INTER_AREA),
                albumentations.OneOf([
                    albumentations.HorizontalFlip(p=0.3),
                    albumentations.ShiftScaleRotate(p=0.3, rotate_limit=90),
                    albumentations.VerticalFlip(p=0.3),
                    albumentations.RandomBrightnessContrast(p=0.3),
                    albumentations.GaussNoise(p=0.3)                    
                ],p=1),
                albumentations.ToTensorV2(),
                albumentations.Normalize()
            ]
        )
        coco = COCO_Dataset(root_dir='/data/coco', set_name='train2014', split='TRAIN', transform=transform)
        data_loader = torch.utils.data.DataLoader(dataset=coco, batch_size=4, shuffle=True, num_workers=1, collate_fn=collate_fn)
        for idx, data in enumerate(data_loader):
                img, bbox, label, seg = data
                break