
import json
from PIL import Image
import torch.utils.data as data
from transformers import  AutoImageProcessor
import torch


class FieldParser:
    def __init__(
            self,
            args
    ):
        super().__init__()
        self.args = args
        self.dataset = args.dataset
        if args.use_siglip:
            self.rad_dino_processor = AutoImageProcessor.from_pretrained(args.siglip_path)
        else:    
            self.rad_dino_processor = AutoImageProcessor.from_pretrained(args.rad_dino_path)


    def _parse_image(self, img):
        pixel_values = self.vit_feature_extractor(img, return_tensors="pt").pixel_values
        return pixel_values[0] 

    # from https://github.com/cuhksz-nlp/R2Gen/blob/main/modules/tokenizers.py


    def parse(self, features):
        to_return = {'id': features['id']}
        to_return['dataset_id'] = 'sn'
        to_return['finding_flag'] = True
        to_return['impression_flag'] = True
        report = ''

        to_return['APPA_flag'] = features['APPA_FLAG']
        to_return['input_text'] = features['report']
        # chest x-ray images

        """
        APPA_imagepath = features['APPA_Imagepath']
        """
        APPA_imagepath = features['id']
        image = Image.open(self.args.base_dir + APPA_imagepath).convert("RGB")
        inputs = self.rad_dino_processor(images=image, return_tensors="pt")
        images = inputs.data['pixel_values']
        to_return["image"] = torch.squeeze(images, 0)  # image tensor
        # 尝试获取 scores，如果不存在或为空，则使用空列表
        scores = features.get('scores', [0.0])
        scores = [float(i) for i in scores]
        scores = torch.tensor(scores, dtype=torch.float32)
        scores = torch.nn.functional.pad(scores, (0, 100 - scores.size(0)), 'constant', 0)
        to_return['scores'] = scores


        indication = features['indication_pure']
        h_i = ''
        if indication != 0:
            h_i = 'INDICATION: ' + indication 
        to_return['h_i'] = h_i
        prior = features.get('prior_study')
        if prior is None:
            # 真正没有过去报告
            to_return['prior_text'] = ""
            to_return['filtered_lastfinding'] = ""
        else:
            # 至少有过一次检查
            sent_labels_combined = []
            for key in ('latest_study', 'second_recent_study'):
                study = prior.get(key)
                if study:
                    sent_labels_combined.extend(study.get('sentences_labels', []))

            # 扁平去重，得到 prior_text
            unique_labels = sorted({lab for grp in sent_labels_combined for lab in grp if lab})
            # 无标签时也要保持空字符串
            to_return['prior_text'] = ", ".join(unique_labels)
            # filtered_lastfinding 直接用 history_label （哪怕只有 “No Finding”）
            filtered = features.get('history_label', [])
            to_return['filtered_lastfinding'] = ", ".join(filtered)

        return to_return

    def transform_with_parse(self, inputs):
        return self.parse(inputs)


class ParseDataset(data.Dataset):
    def __init__(self, args, split='train'):
        self.args = args
        self.meta = json.load(open(args.sn_annotation, 'r'))
        self.meta = self.meta[split]
        self.parser = FieldParser(args)
        self.dataset = args.dataset

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        return self.parser.transform_with_parse(self.meta[index])

def create_datasets_sn(args):
    train_dataset = ParseDataset(args, 'train')
    dev_dataset = ParseDataset(args, 'val')
    test_dataset = ParseDataset(args, 'test')
    return train_dataset,dev_dataset,test_dataset



