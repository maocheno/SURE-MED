
import os
import json
import re
from PIL import Image
import torch.utils.data as data
from transformers import  AutoImageProcessor
import torch
from tqdm import tqdm

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
    def clean_report(self, report):
        # clean Iu-xray reports
        if self.dataset == "iu_xray":
            report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
            sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                            replace('\\', '').replace("'", '').strip().lower())
            tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
            report = ' . '.join(tokens) + ' .'
        # clean MIMIC-CXR reports
        else:
            report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
                .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
                .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
                .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
                .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
                .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
                .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ').replace(':', ' :') \
                .strip().lower().split('. ')
            sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+()\[\]{}]', '', t.replace('"', '').replace('/', '')
                                .replace('\\', '').replace("'", '').strip().lower())
            tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
            report = ' . '.join(tokens) + ' .'
        return report

    def parse(self, features):
        if self.args.iu_xray:
            to_return = {'id': features['id']}
            to_return['dataset_id'] = 'mn'
            to_return['input_text'] = features['report']
            APPA_imagepath = features['APPA_Imagepath']
            image = Image.open(self.args.iu_base_dir + APPA_imagepath).convert("RGB")
            inputs = self.rad_dino_processor(images=image, return_tensors="pt")
            images = inputs.data['pixel_values']
            to_return["image"] = torch.squeeze(images, 0)  # image tensor
            """
            prior = features.get('prior_study', None)
            prior_segment = ""
            to_return['has_prior'] = False
            if prior and prior.get('latest_study'):          
                latest = prior['latest_study']
                facts = latest.get('findings_factual_serialization') or []
                # 过滤并去空
                facts = [f.strip() for f in facts if isinstance(f, str) and f.strip()]
                if facts:
                    prior_segment = "[CLS]" + "[SEP]".join(facts) + "[SEP]"
                imp = (latest.get('impression') or '').strip()
                if imp:
                    prior_segment += f"impression: {imp}[SEP]"
            if prior_segment:
                to_return['has_prior'] = True

            to_return['last_core_findings'] = prior_segment
            """
            to_return['LATERAL_FLAG'] = features['LATERAL_FLAG']
            # chest x-ray images
            lateral_imagepath = features.get('lateral_imagepath', '')
            if lateral_imagepath is None or lateral_imagepath.strip() == '':
                # 2) 用全零张量做占位，shape 跟 frontal image 一致
                #    假设你前面已经做过 frontal image 处理：
                #    ap_image = to_return['APPA_image']  # [3, H, W]
                C, H, W = to_return['image'].shape
                placeholder = torch.zeros((C, H, W), dtype=torch.float32)
                to_return['lateral_image'] = placeholder

            else:
                # 正常读入侧面图
                img = Image.open(os.path.join(self.args.base_dir, lateral_imagepath)).convert('RGB')
                lateral_inputs = self.rad_dino_processor(images=img, return_tensors="pt")
                lateral_images = lateral_inputs.pixel_values  # [1, C, H, W]
                to_return['lateral_image'] = lateral_images.squeeze(0)

            prior = features.get('prior_study', None)
            prior_segment = ""

            if prior and 'latest_study' in prior:
                latest = prior['latest_study']
                sentences_labels = latest.get('sentences_labels', [])

                # 展平所有标签并去重
                unique_labels = set()
                for labels in sentences_labels:
                    for label in labels:
                        unique_labels.add(label)

                if unique_labels:
                    prior_segment = ",".join(sorted(unique_labels))

            to_return['prior_text'] = prior_segment
            
            hc = features.get('high_conf_labels', [])
            if not hc:
                hc = ["No Finding"]
            # 去重并排序（可选），再拼成一句话
            unique_hc = sorted(set(hc))
            to_return['image_check'] = ", ".join(unique_hc)

            h_i = ''

            to_return['h_i'] = h_i
        else:
            to_return = {'id': features['id']}
            to_return['dataset_id'] = 'mn'
            to_return['input_text'] = features['report']
            to_return['APPA_flag'] = features['APPA_FLAG']
            # chest x-ray images
            APPA_imagepath = features['APPA_Imagepath']
            image = Image.open(self.args.base_dir + APPA_imagepath).convert("RGB")
            inputs = self.rad_dino_processor(images=image, return_tensors="pt")
            images = inputs.data['pixel_values']
            to_return["image"] = torch.squeeze(images, 0)  # image tensor
            """
            prior = features.get('prior_study', None)
            prior_segment = ""
            to_return['has_prior'] = False
            if prior and prior.get('latest_study'):          
                latest = prior['latest_study']
                facts = latest.get('findings_factual_serialization') or []
                # 过滤并去空
                facts = [f.strip() for f in facts if isinstance(f, str) and f.strip()]
                if facts:
                    prior_segment = "[CLS]" + "[SEP]".join(facts) + "[SEP]"
                imp = (latest.get('impression') or '').strip()
                if imp:
                    prior_segment += f"impression: {imp}[SEP]"
            if prior_segment:
                to_return['has_prior'] = True

            to_return['last_core_findings'] = prior_segment
            """
            to_return['LATERAL_FLAG'] = features['LATERAL_FLAG']
            # chest x-ray images
            lateral_imagepath = features.get('lateral_imagepath', '')
            if lateral_imagepath is None or lateral_imagepath.strip() == '':
                # 2) 用全零张量做占位，shape 跟 frontal image 一致
                #    假设你前面已经做过 frontal image 处理：
                #    ap_image = to_return['APPA_image']  # [3, H, W]
                C, H, W = to_return['image'].shape
                placeholder = torch.zeros((C, H, W), dtype=torch.float32)
                to_return['lateral_image'] = placeholder

            else:
                # 正常读入侧面图
                img = Image.open(os.path.join(self.args.base_dir, lateral_imagepath)).convert('RGB')
                lateral_inputs = self.rad_dino_processor(images=img, return_tensors="pt")
                lateral_images = lateral_inputs.pixel_values  # [1, C, H, W]
                to_return['lateral_image'] = lateral_images.squeeze(0)

            scores = features.get('scores', [0.0])
            scores = [float(i) for i in scores]
            scores = torch.tensor(scores, dtype=torch.float32)
            scores = torch.nn.functional.pad(scores, (0, 100 - scores.size(0)), 'constant', 0)
            to_return['scores'] = scores
            # 假设 features 是当前记录的 dict，to_return 是你要构造的输出 dict
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

            indication = features['indication_pure']
            h_i = ''
            if indication != 0:
                h_i = 'INDICATION: ' + indication 
            to_return['h_i'] = h_i

        return to_return

    def transform_with_parse(self, inputs):
        return self.parse(inputs)


class ParseDataset(data.Dataset):
    def __init__(self, args, split='train'):
        self.args = args
        if args.iu_xray:
            self.meta = json.load(open(args.iu_xray_annotation, 'r'))
        else:
            self.meta = json.load(open(args.mn_annotation, 'r'))
        self.meta = self.meta[split]
        self.parser = FieldParser(args)
        self.dataset = args.dataset

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        return self.parser.transform_with_parse(self.meta[index])

def create_datasets_mn(args):
    train_dataset = ParseDataset(args, 'train')
    dev_dataset = ParseDataset(args, 'val')
    test_dataset = ParseDataset(args, 'test')
    return train_dataset,dev_dataset,test_dataset



