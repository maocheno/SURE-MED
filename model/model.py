import os
import json
import torch
import torch.nn as nn
import lightning.pytorch as pl
from transformers import LlamaForCausalLM, LlamaTokenizer
from evalcap.bleu.bleu import Bleu
from evalcap.rouge.rouge import Rouge
from evalcap.cider.cider import Cider
from evalcap.meteor.meteor import Meteor
from peft import get_peft_model, LoraConfig, TaskType

import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from transformers import AutoImageProcessor
from models.chexbert import CheXbert
import numpy as np
import pandas as pd
from models.metrics import compute_mlc
from transformers import get_cosine_schedule_with_warmup

class Focus_Med(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)

        full_siglip  = AutoModel.from_pretrained(args.siglip_path)
        self.visual_encoder = full_siglip.vision_model

        if args.freeze_vm:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            print(f'Loading Frozen vision encoder:{args.rad_dino_path} -- Done')

        if args.stage_class == 1:
            self.APPA_visual_query_tokens = nn.Parameter(
                    torch.zeros(1, args.visual_token_number, 1024)
                )
            self.APPA_visual_query_tokens.data.normal_(mean=0.0, std=0.02)
            self.APPA_crossattention_block = nn.MultiheadAttention(1024, 16, dropout=0.1, add_bias_kv=True,
                                                                       add_zero_attn=True)
            self.APPA_layernorm_768_1 = nn.LayerNorm(1024)
            self.APPA_llama_proj_1 = nn.Linear(1024, 1024)
            self.APPA_layernorm_768_2 = nn.LayerNorm(1024)
            self.APPA_llama_proj_2 = nn.Linear(1024, 2048)
            self.APPA_llama_proj_3 = nn.Linear(2048, 4096)
            self.APPA_layernorm_4096_1 = nn.LayerNorm(4096)         
        else:
            self.APPA_visual_query_tokens = nn.Parameter(
                    torch.zeros(1, args.visual_token_number, 1024)
                )
            self.APPA_visual_query_tokens.data.normal_(mean=0.0, std=0.02)
            self.APPA_crossattention_block = nn.MultiheadAttention(1024, 16, dropout=0.1, add_bias_kv=True,
                                                                       add_zero_attn=True)
            self.APPA_layernorm_768_1 = nn.LayerNorm(1024)
            self.APPA_llama_proj_1 = nn.Linear(1024, 1024)
            self.APPA_layernorm_768_2 = nn.LayerNorm(1024)
            self.APPA_llama_proj_2 = nn.Linear(1024, 2048)
            self.APPA_llama_proj_3 = nn.Linear(2048, 4096)
            self.APPA_layernorm_4096_1 = nn.LayerNorm(4096)

            self.lateral_crossattention_block = nn.MultiheadAttention(1024, 16, dropout=0.1, add_bias_kv=True,
                                                                       add_zero_attn=True)
            self.lateral_layernorm_768_1 = nn.LayerNorm(1024)
            self.lateral_llama_proj_1 = nn.Linear(1024, 1024)
            self.lateral_layernorm_768_2 = nn.LayerNorm(1024)
            self.lateral_llama_proj_2 = nn.Linear(1024, 2048)
            self.lateral_llama_proj_3 = nn.Linear(2048, 4096)
            self.lateral_layernorm_4096_1 = nn.LayerNorm(4096)

            self.iit_proj = nn.Linear(4096 * 2,4096)
            self.iit_layernorm = nn.LayerNorm(4096)

        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(args.vicuna_model, use_fast=False)
        self.llama_tokenizer.pad_token_id = 0
        if args.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                args.vicuna_model,
                torch_dtype=torch.bfloat16,
                load_in_8bit=True,
                device_map="auto"
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                args.vicuna_model,
                torch_dtype=torch.bfloat16,
            )

        if args.llm_use_lora:
            self.embed_tokens = self.llama_model.get_input_embeddings()
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=args.llm_r,
                lora_alpha=args.llm_alpha,
                # target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
                lora_dropout=args.lora_dropout,
                bias='none',
            )
            self.llama_model = get_peft_model(self.llama_model, peft_config)
            self.llama_model.print_trainable_parameters()
            print('Loading LLAMA LoRA Done')
        else:
            self.embed_tokens = self.llama_model.get_input_embeddings()
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
            print('Loading LLAMA Done')

        self.end_sym = args.end_sym
        self.f_prompt = 'Write radiological finding section of report for current chest X-ray visual feature'


        self.val_step_outputs = []
        self.val_sn_step_outputs = []
        self.val_mn_step_outputs = []
        self.test_step_outputs = []
        self.val_score = 0.0

        visual_state_dict = self.visual_encoder.state_dict()
        if args.visual_delta_file is not None:
            state_dict = \
            torch.load(args.visual_delta_file, map_location=torch.device(f'cuda:{torch.cuda.current_device()}'))[
                'model']
            self.load_state_dict(state_dict=state_dict, strict=False)
            print(f'Load checkpoint from {args.visual_delta_file}')
        if args.delta_file is not None:
            state_dict = torch.load(args.delta_file, map_location=torch.device(f'cuda:{torch.cuda.current_device()}'))[
                'model']
            self.load_state_dict(state_dict=state_dict, strict=False)
            print(f'Load checkpoint from {args.delta_file}')

    def score(self, ref, hypo):
        """
        ref, dictionary of reference sentences (id, sentence)
        hypo, dictionary of hypothesis sentences (id, sentence)
        score, dictionary of scores
        """
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(), "ROUGE_L"),
            (Meteor(), "METEOR"),
            (Cider(), "CIDEr")
        ]
        final_scores = {}
        for scorer, method in scorers:
            score, scores = scorer.compute_score(ref, hypo)
            if type(score) == list:
                for m, s in zip(method, score):
                    final_scores[m] = s
            else:
                final_scores[method] = score
        return final_scores

    def encode_APPA_img(self, images):
        outputs = self.visual_encoder(pixel_values=images).last_hidden_state
        outputs = outputs[:,1:,:]
        vision_query_tokens = self.APPA_visual_query_tokens.expand(images.shape[0], -1, -1)
        inputs_llama = self.APPA_crossattention_block(vision_query_tokens.transpose(0, 1),
                                                         outputs.transpose(0, 1),
                                                         outputs.transpose(0, 1))[0].transpose(0, 1)
        inputs_llama = self.APPA_layernorm_768_1(inputs_llama)
        inputs_llama1 = self.APPA_llama_proj_1(inputs_llama)
        inputs_llama1 = self.APPA_layernorm_768_2(inputs_llama1) + inputs_llama

        inputs_llama2 = self.APPA_llama_proj_2(inputs_llama1)
        inputs_llama2 = F.gelu(inputs_llama2)
        inputs_llama2 = self.APPA_llama_proj_3(inputs_llama2)
        inputs_llama2 = self.APPA_layernorm_4096_1(inputs_llama2)
        atts_llama = torch.ones(inputs_llama2.size()[:-1], dtype=torch.long).to(images.device)
        return inputs_llama2, atts_llama, inputs_llama1

    def encode_lateral_image(self, images, query_feature):
        outputs = self.visual_encoder(images).last_hidden_state
        outputs = outputs[:,1:,:]
        inputs_llama = self.lateral_crossattention_block(query_feature.transpose(0, 1),
                                                         outputs.transpose(0, 1),
                                                         outputs.transpose(0, 1))[0].transpose(0, 1)
        inputs_llama = self.lateral_layernorm_768_1(inputs_llama)
        inputs_llama1 = self.lateral_llama_proj_1(inputs_llama)
        inputs_llama1 = self.lateral_layernorm_768_2(inputs_llama1) + inputs_llama
        inputs_llama1 = self.lateral_llama_proj_2(inputs_llama1)
        inputs_llama1 = F.gelu(inputs_llama1)
        inputs_llama1 = self.lateral_llama_proj_3(inputs_llama1)
        inputs_llama1 = self.lateral_layernorm_4096_1(inputs_llama1)
        atts_llama = torch.ones(inputs_llama1.size()[:-1], dtype=torch.long).to(images.device)
        return inputs_llama1, atts_llama


    def prompt_sn(self, img_embeds, atts_img, samples):
        batch_size = img_embeds.shape[0]
        prompt = []
        for b in range(batch_size):
            h_i = samples['h_i'][b]
            prior_text = samples['prior_text'][b] if 'prior_text' in samples else ""
            self.args.iu_xray = True
            if self.args.iu_xray:
                    prompt_text = (
                    f'Human: There is no previous report or historical information available.\n'
                    f'Do not mention or speculate about any past findings, changes over time, or comparisons with prior studies.\n'
                    f'Current chest X-ray image feature:\n <Img><ImageHere></Img> \n'
                    f'{self.f_prompt}. \nAssistant:'
                )
            else:
                filtered_lastfinding = samples['filtered_lastfinding'][b]
                if prior_text.strip():
                    prompt_text = (
                        f'Human: {h_i}\n'
                        f'Original Last findings: {prior_text}\n'
                        f"Relevant prior findings (filtered):{filtered_lastfinding}\n"
                        f'Current chest X-ray image feature:\n <Img><ImageHere></Img> \n'
                        f'{self.f_prompt}. \nAssistant:'
                    )
                else:
                    prompt_text = (
                        f'Human: {h_i}\n'
                        f'Current chest X-ray image feature:\n <Img><ImageHere></Img> \n'
                        f'{self.f_prompt}. \nAssistant:'
                    )
                #print(prompt_text)
            prompt.append(prompt_text)

        p_after = []
        p_before = []
        for p in prompt:
            p_before_tmp, p_after_tmp = p.split('<ImageHere>')
            p_after.append(p_after_tmp)
            p_before.append(p_before_tmp)
        # p_before = f'Human: <Img>'
        pad_size = self.llama_tokenizer.padding_side
        self.llama_tokenizer.padding_side = 'left'
        self.llama_tokenizer.truncation_side = 'right'   # truncate history
        p_before_tokens = self.llama_tokenizer(
            p_before,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=100,
            add_special_tokens=False
        ).to(img_embeds.device)
        self.llama_tokenizer.padding_side = 'right'
        p_after_tokens = self.llama_tokenizer(
            p_after, return_tensors="pt", add_special_tokens=False, padding=True).to(img_embeds.device)
        self.llama_tokenizer.padding_side = pad_size
        p_before_embeds = self.embed_tokens(p_before_tokens.input_ids)
        p_after_embeds = self.embed_tokens(p_after_tokens.input_ids)  # .expand(batch_size, -1, -1)

        post_pad_length = torch.logical_not(p_before_tokens.attention_mask).sum(-1)
        new_p_before_embeds = torch.zeros(batch_size, p_before_embeds.size(1) + 1, 4096).type(img_embeds.dtype).to(img_embeds.device)
        new_p_before_mask = torch.zeros(batch_size, p_before_embeds.size(1) + 1).type(atts_img.dtype).to(img_embeds.device)
        for b in range(batch_size):
            post_pad_len = post_pad_length[b]
            new_p_before_embeds[b,:post_pad_len,:] = p_before_embeds[b,:post_pad_len,:]
            bos = torch.ones([1, 1],dtype=p_before_tokens.input_ids.dtype,device=p_before_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
            bos_embeds = self.embed_tokens(bos)
            new_p_before_embeds[b,post_pad_len,:] = bos_embeds
            new_p_before_mask[b,post_pad_len:] = 1
            new_p_before_embeds[b,(post_pad_len+1):,:] = p_before_embeds[b,post_pad_len:,:]

        wrapped_img_embeds = torch.cat([new_p_before_embeds, img_embeds, p_after_embeds], dim=1)
        wrapped_atts_img = torch.cat([new_p_before_mask, atts_img, p_after_tokens.attention_mask], dim=1)
        return wrapped_img_embeds, wrapped_atts_img

    def forward(self, samples):
        image = samples["image"]
        dataset_id = samples['dataset_id'][0]
        print(dataset_id)

        def all_elements_equal(lst):
            return all(x == lst[0] for x in lst)

        lst = samples['dataset_id']
        if all_elements_equal(lst) == False:
            print(all_elements_equal(lst))
        self.llama_tokenizer.padding_side = "right"
        text = [t + self.end_sym for t in samples["input_text"]]
        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=True
        ).to(image.device)
        to_regress_embeds = self.embed_tokens(to_regress_tokens.input_ids)

        img_embeds = ''
        atts_img = ''
        if dataset_id == 'sn':
            img_embeds1, atts_img1, query_feature = self.encode_APPA_img(image)
            if self.args.stage_class == 1:
                batch_size = img_embeds1.shape[0]
                bos = torch.ones([batch_size, 1],
                                 dtype=to_regress_tokens.input_ids.dtype,
                                 device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
                bos_embeds = self.embed_tokens(bos)
                atts_bos = atts_img1[:, :1]
                img_embeds = torch.cat([bos_embeds,img_embeds1], dim=1)
                atts_img = torch.cat([atts_bos,atts_img1], dim=1)
            if self.args.stage_class != 1:
                zero_padding = torch.zeros(img_embeds1.size(0), img_embeds1.size(1), 4096, device=img_embeds1.device)
                new_embeds = torch.cat([img_embeds1, zero_padding], dim=2)
                new_embeds = self.iit_proj(new_embeds)
                new_embeds = self.iit_layernorm(new_embeds)
                img_embeds, atts_img = self.prompt_sn(new_embeds, atts_img1, samples)

        if dataset_id == 'mn':
            # 1) APPA 部分
            APPA_image = samples["image"]                                            # [B,3,H,W]
            img_embeds1, atts_img1, query_feature = self.encode_APPA_img(APPA_image) # img_embeds1: [B, T, D]

            # 2) lateral 部分，根据 flag 分两路
            lateral_flags = samples['LATERAL_FLAG']  # Tensor of shape [B], dtype=torch.bool
            lateral_image  = samples['lateral_image'] 

            B, T, D = img_embeds1.shape
            # 全零占位：假设 lateral_embed 也应该是 [B, T, D]
            lateral_embed = torch.zeros((B, T, D), dtype=img_embeds1.dtype, device=img_embeds1.device)
            atts_lateral  = torch.zeros((B, T),     dtype=atts_img1.dtype, device=atts_img1.device)

            # 只有那些有侧面图的样本要跑 encode_lateral_image
            if lateral_flags.any():
                idx = torch.nonzero(lateral_flags, as_tuple=True)[0]  # indices of True
                lat_imgs = lateral_image[idx]                         # [N,3,H,W]
                qf       = query_feature[idx]                         # [N, T_q, D_q]
                lat_emb, lat_att = self.encode_lateral_image(lat_imgs, qf)
                # 把结果写回对应位置
                lateral_embed[idx] = lat_emb
                atts_lateral[idx]  = lat_att

            # 3) 拼接：img_embeds1 + lateral_embed + padding

            new_embeds   = torch.cat([img_embeds1, lateral_embed], dim=2)  # [B, T, D*2]

            new_embeds   = self.iit_proj(new_embeds)
            new_embeds   = self.iit_layernorm(new_embeds)
            img_embeds, atts_img = self.prompt_sn(new_embeds, atts_img1, samples)

        targets = torch.zeros_like(to_regress_tokens.attention_mask).long().fill_(-100)

        # only apply loss to answer tokens
        targets_idx = to_regress_tokens.attention_mask.bool()
        targets[:, -targets_idx.shape[1]:][targets_idx] = to_regress_tokens.input_ids[targets_idx]
        targets[:, -targets_idx.shape[1]] = -100

        empty_targets = (
            torch.ones([atts_img.shape[0], atts_img.shape[1]],
                       dtype=torch.long).to(image[0].device).fill_(-100)  # plus one for bos
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        inputs_embeds = torch.cat([img_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_img, to_regress_tokens.attention_mask], dim=1)

        device = inputs_embeds.device
        length = attention_mask.size(1)
        zero_tensor = torch.zeros(attention_mask.size(0), length - 100, device=device)
        scores_ori = samples['scores']
        scores = torch.cat([zero_tensor, scores_ori], dim=1)

        outputs    = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=False,
            labels=None,               # 先不要 labels，让我们自己算 loss
        )
        logits     = outputs.logits  # [B, T, V]

        # 2) shift 对齐 “predict next token”
        shift_logits = logits[:, :-1, :].contiguous()   # [B, T-1, V]
        shift_labels = targets[:, 1:].contiguous()      # [B, T-1]
        shift_mask   = attention_mask[:, 1:].contiguous()  # [B, T-1]

        # 3) flatten 并掐掉 padding
        B, L = shift_labels.size()
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))  # [B*(T-1), V]
        # 3) flatten
        flat_labels = shift_labels.view(-1)                  # [B*(T-1)]
        flat_mask   = shift_mask.view(-1).bool()             # [B*(T-1)]

        # 新增：HF 内部是用 ignore_index=-100 跳过 label=-100 的位置
        ignore_mask = flat_labels != -100                    # [B*(T-1)], True 表示要算 loss

        # 合并两个 mask：只保留既非 padding 又 label ≠ -100 的 token
        valid_mask = flat_mask & ignore_mask                 # [B*(T-1)]

        # 之后取真实 token
        valid_logits = flat_logits[valid_mask]
        valid_labels = flat_labels[valid_mask]

        # 4) 用 CrossEntropyLoss(reduction='none') 拿到每个 token 的 loss
        loss_fct  = torch.nn.CrossEntropyLoss(reduction='none')
        loss_flat = loss_fct(valid_logits, valid_labels)           # [N_valid]

        # 5) 同样对 scores 做 shift/flatten/mask
        shift_scores = scores[:, 1:].contiguous().view(-1)     # [B*(T-1)]
        scores_flat  = shift_scores[valid_mask]                     # [N_valid]

        # 6) 归一化并映射到 [α, 1.0]
        α            = 0.1
        max_w        = scores_flat.max().clamp(min=1.0)            # 比如 2.0
        scores_norm  = scores_flat / max_w                         # [0,1]
        soft_scores  = α + (1.0 - α) * scores_norm                  # [α,1]

        # 7) 计算两路 mean loss
        average_loss = loss_flat.mean()
        key_loss     = (loss_flat * soft_scores).mean()

        total_loss = average_loss + self.args.sentence_ratio * key_loss
        if self.args.loss_mode == 'None':
            total_loss = average_loss
        self.log("average_loss", average_loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("key_loss", key_loss,         prog_bar=True, on_step=True, on_epoch=False)
        return {"loss": total_loss}

    def training_step(self, batch, batch_idx):
        result = self(batch)
        self.log_dict(result, prog_bar=True)
        return result

    def save_checkpoint(self, eval_res, chexbert_f1, recall_micro):
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        param_grad_dic = {
            k: v.requires_grad for (k, v) in self.named_parameters() if v.requires_grad
        }
        state_dict = self.state_dict()
        for k in list(state_dict.keys()):
            if k not in param_grad_dic.keys():
                del state_dict[k]
        save_obj = {
            "model": state_dict,
            "config": self.hparams,
            "epoch": current_epoch,
            "step": global_step
        }
        os.makedirs(os.path.join(self.hparams.savedmodel_path, 'pths'), exist_ok=True)
        save_to = os.path.join(
            self.hparams.savedmodel_path, 'pths',
            "checkpoint_epoch{}_step{}_bleu{:3f}_cider{:3f}_chexbert{:3f}_recall{:3f}.pth".format(
                current_epoch, global_step, eval_res['Bleu_4'],
                eval_res['CIDEr'], chexbert_f1, recall_micro
            ),
        )
        self.print("Saving checkpoint at step {} to {}.".format(global_step, save_to))
        torch.save(save_obj, save_to)


    def validation_step(self, samples, batch_idx):
        self.llama_tokenizer.padding_side = "right"
        to_regress_tokens = self.llama_tokenizer(
            samples['input_text'],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        )
        image = samples["image"]
        dataset_id = samples['dataset_id'][0]
        img_embeds = ''
        atts_img = ''
        print(dataset_id)
        if dataset_id == 'sn':
            img_embeds1, atts_img1, query_feature = self.encode_APPA_img(image)
            embed_device = self.embed_tokens.weight.device
            if self.args.stage_class == 1:
                batch_size = img_embeds1.shape[0]
                bos = torch.ones([batch_size, 1],
                                 dtype=to_regress_tokens.input_ids.dtype,
                                 device=embed_device) * self.llama_tokenizer.bos_token_id
                bos_embeds = self.embed_tokens(bos)
                atts_bos = atts_img1[:, :1]
                img_embeds = torch.cat([bos_embeds,img_embeds1], dim=1)
                atts_img = torch.cat([atts_bos,atts_img1], dim=1)

        if dataset_id == 'mn':
            # 1) APPA 部分
            APPA_image = samples["image"]                                            # [B,3,H,W]
            img_embeds1, atts_img1, query_feature = self.encode_APPA_img(APPA_image) # img_embeds1: [B, T, D]

            # 2) lateral 部分，根据 flag 分两路
            lateral_flags = samples['LATERAL_FLAG']  # Tensor of shape [B], dtype=torch.bool
            lateral_image  = samples['lateral_image'] 

            B, T, D = img_embeds1.shape
            # 全零占位：假设 lateral_embed 也应该是 [B, T, D]
            lateral_embed = torch.zeros((B, T, D), dtype=img_embeds1.dtype, device=img_embeds1.device)
            atts_lateral  = torch.zeros((B, T),     dtype=atts_img1.dtype, device=atts_img1.device)

            # 只有那些有侧面图的样本要跑 encode_lateral_image
            if lateral_flags.any():
                idx = torch.nonzero(lateral_flags, as_tuple=True)[0]  # indices of True
                lat_imgs = lateral_image[idx]                         # [N,3,H,W]
                qf       = query_feature[idx]                         # [N, T_q, D_q]
                lat_emb, lat_att = self.encode_lateral_image(lat_imgs, qf)
                # 把结果写回对应位置
                lateral_embed[idx] = lat_emb
                atts_lateral[idx]  = lat_att

            # 3) 拼接：img_embeds1 + lateral_embed + padding

            new_embeds   = torch.cat([img_embeds1, lateral_embed], dim=2)  # [B, T, D*2]

            new_embeds   = self.iit_proj(new_embeds)
            new_embeds   = self.iit_layernorm(new_embeds)
            img_embeds, atts_img = self.prompt_sn(new_embeds, atts_img1, samples)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=atts_img.dtype,
                         device=atts_img.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, -1:]

        inputs_embeds = torch.cat([img_embeds, bos_embeds], dim=1)
        attention_mask = torch.cat([atts_img, atts_bos], dim=1)

        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask = attention_mask,
            num_beams=self.hparams.beam_size,
            do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
            temperature=self.hparams.temperature,
        )
        hypo = [self.decode(i) for i in outputs]
        ref = [self.decode(i) for i in to_regress_tokens['input_ids']]
        self.val_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})
        if dataset_id == 'sn':
            self.val_sn_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})
        if dataset_id == 'mn':
            self.val_mw_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})
        return hypo, ref

    def decode(self, output_token):
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        output_text = self.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('</s>')[0].strip()
        output_text = output_text.replace('<unk>', '')
        return output_text

    def calculate_metric(self,val_step_outputs):
        ref, hypo, ids = [], [], []
        for i in val_step_outputs:
            ref.extend(i['ref'])
            hypo.extend(i['hypo'])
            ids.extend(i['id'])

        # chexbert
        device = torch.device('cuda:0')
        chexbert = CheXbert(
            ckpt_dir='./hf',
            bert_path=self.hparams.bert_path,
            checkpoint_path=self.hparams.chexbert_path,
            device=device
        ).to(device)

        # calculate chexbert
        ref_pred_list = []
        hypo_pred_list = []
        for index in range(len(ref)):
            refs = [ref[index]]
            hypos = [hypo[index]]
            ref_pred = chexbert(refs).squeeze().cpu().numpy()
            hypo_pred = chexbert(hypos).squeeze().cpu().numpy()
            ref_pred_list.append(ref_pred)
            hypo_pred_list.append(hypo_pred)

        ref_pred_list = np.array(ref_pred_list)
        df = pd.DataFrame(ref_pred_list, columns=[f'feature_{i + 1}' for i in range(14)])
        df.insert(0, 'id', ids)
        df.replace(0, np.nan, inplace=True)  # blank class is NaN
        df.replace(3, -1, inplace=True)  # uncertain class is -1
        df.replace(2, 0, inplace=True)  # negative class is 0
        gts_path = os.path.join(self.args.savedmodel_path, 'ref_labeled_reports.csv')
        df.to_csv(gts_path, index=False)

        hypo_pred_list = np.array(hypo_pred_list)
        df = pd.DataFrame(hypo_pred_list, columns=[f'feature_{i + 1}' for i in range(14)])
        df.insert(0, 'id', ids)
        df.replace(0, np.nan, inplace=True)  # blank class is NaN
        df.replace(3, -1, inplace=True)  # uncertain class is -1
        df.replace(2, 0, inplace=True)  # negative class is 0
        res_path = os.path.join(self.args.savedmodel_path, 'pred_labeled_reports.csv')
        df.to_csv(res_path, index=False)

        res_data, gts_data = pd.read_csv(res_path), pd.read_csv(gts_path)
        res_data, gts_data = res_data.fillna(0), gts_data.fillna(0)

        label_set = res_data.columns[1:].tolist()
        res_data, gts_data = res_data.iloc[:, 1:].to_numpy(), gts_data.iloc[:, 1:].to_numpy()
        res_data[res_data == -1] = 1
        gts_data[gts_data == -1] = 1
        chexbert_f1_MACRO = 0
        if len(ref) > 100:
            metrics = compute_mlc(gts_data, res_data, label_set)


        ref = {k: [v] for k, v in zip(ids, ref)}
        hypo = {k: [v] for k, v in zip(ids, hypo)}
        eval_res = self.score(ref=ref, hypo=hypo)
        self.log_dict(eval_res, sync_dist=True, logger=True)

        result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
        os.makedirs(result_folder, exist_ok=True)
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        json.dump(hypo, open(os.path.join(result_folder, f"result_{current_epoch}_{global_step}" + '.json'), 'w'))
        json.dump(ref, open(os.path.join(result_folder, 'refs.json'), 'w'))
        return metrics,eval_res

    def on_validation_epoch_end(self):

        if self.args.stage_class == 2:
            mn_ce, mn_eval = self.calculate_metric(self.val_step_outputs)
            print(mn_ce)
            print(mn_eval)
            if self.trainer.local_rank == 0:
                # if val_score > self.val_score:
                self.save_checkpoint(mn_eval, mn_ce['F1_MICRO'],mn_ce['RECALL_MICRO'])
        if self.args.stage_class == 1:
            sn_ce, sn_eval = self.calculate_metric(self.val_step_outputs)
            print(sn_ce)
            print(sn_eval)
            if self.trainer.local_rank == 0:
                # if val_score > self.val_score:
                self.save_checkpoint(sn_eval, sn_ce['F1_MICRO'],sn_ce['RECALL_MICRO'])
        self.val_step_outputs.clear()
        self.val_sn_step_outputs.clear()
        self.val_mn_step_outputs.clear()

    def test_step(self, samples, batch_idx):
        dataset_id = samples['dataset_id'][0]
        print(dataset_id)

        def all_elements_equal(lst):
            return all(x == lst[0] for x in lst)

        lst = samples['dataset_id']
        if all_elements_equal(lst) == False:
            print(all_elements_equal(lst))
        self.llama_tokenizer.padding_side = "right"
        to_regress_tokens = self.llama_tokenizer(
            samples['input_text'],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=500,  #self.hparams.max_length
            add_special_tokens=False
        )
        # .to(image[0].device)
        image = samples["image"]
        dataset_id = samples['dataset_id'][0]
        img_embeds = ''
        atts_img = ''
        if dataset_id == 'sn':
            img_embeds1, atts_img1, query_feature = self.encode_APPA_img(image)
            if self.args.stage_class == 1:
                batch_size = img_embeds1.shape[0]
                bos = torch.ones([batch_size, 1],
                                 dtype=to_regress_tokens.input_ids.dtype,
                                 device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
                bos_embeds = self.embed_tokens(bos)
                atts_bos = atts_img1[:, :1]
                img_embeds = torch.cat([bos_embeds,img_embeds1], dim=1)
                atts_img = torch.cat([atts_bos,atts_img1], dim=1)
            if self.args.stage_class != 1:
                zero_padding = torch.zeros(img_embeds1.size(0), img_embeds1.size(1), 8192, device=img_embeds1.device)
                new_embeds = torch.cat([img_embeds1, zero_padding], dim=2)
                new_embeds = self.iit_proj(new_embeds)
                new_embeds = self.iit_layernorm(new_embeds)
                img_embeds, atts_img = self.prompt_sn(new_embeds, atts_img1, samples)

        if dataset_id == 'mn':
            # 1) APPA 部分
            APPA_image = samples["image"]                                            # [B,3,H,W]
            img_embeds1, atts_img1, query_feature = self.encode_APPA_img(APPA_image) # img_embeds1: [B, T, D]

            # 2) lateral 部分，根据 flag 分两路
            lateral_flags = samples['LATERAL_FLAG']  # Tensor of shape [B], dtype=torch.bool
            lateral_image  = samples['lateral_image'] 

            B, T, D = img_embeds1.shape
            # 全零占位：假设 lateral_embed 也应该是 [B, T, D]
            lateral_embed = torch.zeros((B, T, D), dtype=img_embeds1.dtype, device=img_embeds1.device)
            atts_lateral  = torch.zeros((B, T),     dtype=atts_img1.dtype, device=atts_img1.device)

            # 只有那些有侧面图的样本要跑 encode_lateral_image
            if lateral_flags.any():
                idx = torch.nonzero(lateral_flags, as_tuple=True)[0]  # indices of True
                lat_imgs = lateral_image[idx]                         # [N,3,H,W]
                qf       = query_feature[idx]                         # [N, T_q, D_q]
                lat_emb, lat_att = self.encode_lateral_image(lat_imgs, qf)
                # 把结果写回对应位置
                lateral_embed[idx] = lat_emb
                atts_lateral[idx]  = lat_att

            # 3) 拼接：img_embeds1 + lateral_embed + padding

            new_embeds   = torch.cat([img_embeds1, lateral_embed], dim=2)  # [B, T, D*2]

            new_embeds   = self.iit_proj(new_embeds)
            new_embeds   = self.iit_layernorm(new_embeds)
            img_embeds, atts_img = self.prompt_sn(new_embeds, atts_img1, samples)

           
        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=atts_img.dtype,
                         device=atts_img.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, -1:]

        inputs_embeds = torch.cat([img_embeds, bos_embeds], dim=1)
        attention_mask = torch.cat([atts_img, atts_bos], dim=1)

        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask = attention_mask,
            num_beams=self.hparams.beam_size,
            do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
            temperature=self.hparams.temperature,
        )
        hypo = [self.decode(i) for i in outputs]
        ref = [self.decode(i) for i in to_regress_tokens['input_ids']]
        self.test_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})
        return hypo, ref

    def on_test_epoch_end(self):
        """
        This function is called at the end of the test epoch.
        It is recommended to test on single device to ensure each sample/batch gets evaluated exactly once. This is helpful to make sure benchmarking for research papers is done the right way. Otherwise, in a multi-device setting, samples could occur duplicated when DistributedSampler is used, for eg. with strategy="ddp". It replicates some samples on some devices to make sure all devices have same batch size in case of uneven inputs.
        """
        device = torch.device('cuda:0')
        chexbert = CheXbert(
            ckpt_dir='./hf',
            bert_path=self.hparams.bert_path,
            checkpoint_path=self.hparams.chexbert_path,
            device=device
        ).to(device)

        ref, hypo, ids = [], [], []
        for i in self.test_step_outputs:
            ref.extend(i['ref'])
            hypo.extend(i['hypo'])
            ids.extend(i['id'])

        # calculate chexbert
        ref_pred_list = []
        hypo_pred_list = []
        for index in range(len(ref)):
            refs = [ref[index]]
            hypos = [hypo[index]]
            ref_pred = chexbert(refs).squeeze().cpu().numpy()
            hypo_pred = chexbert(hypos).squeeze().cpu().numpy()
            ref_pred_list.append(ref_pred)
            hypo_pred_list.append(hypo_pred)

        ref_pred_list = np.array(ref_pred_list)
        df = pd.DataFrame(ref_pred_list, columns=[f'feature_{i + 1}' for i in range(14)])
        df.insert(0, 'id', ids)
        df.replace(0, np.nan, inplace=True)  # blank class is NaN
        df.replace(3, -1, inplace=True)  # uncertain class is -1
        df.replace(2, 0, inplace=True)  # negative class is 0
        gts_path = os.path.join(self.args.savedmodel_path, 'ref_labeled_reports.csv')
        df.to_csv(gts_path, index=False)

        hypo_pred_list = np.array(hypo_pred_list)
        df = pd.DataFrame(hypo_pred_list, columns=[f'feature_{i + 1}' for i in range(14)])
        df.insert(0, 'id', ids)
        df.replace(0, np.nan, inplace=True)  # blank class is NaN
        df.replace(3, -1, inplace=True)  # uncertain class is -1
        df.replace(2, 0, inplace=True)  # negative class is 0
        res_path = os.path.join(self.args.savedmodel_path, 'pred_labeled_reports.csv')
        df.to_csv(res_path, index=False)

        res_data, gts_data = pd.read_csv(res_path), pd.read_csv(gts_path)
        res_data, gts_data = res_data.fillna(0), gts_data.fillna(0)

        label_set = res_data.columns[1:].tolist()
        res_data, gts_data = res_data.iloc[:, 1:].to_numpy(), gts_data.iloc[:, 1:].to_numpy()
        res_data[res_data == -1] = 1
        gts_data[gts_data == -1] = 1
        metrics = compute_mlc(gts_data, res_data, label_set)
        chexbert_f1_MICRO = metrics['F1_MICRO']
        print('chexbert_f1_MICRO: ')
        print(chexbert_f1_MICRO)
        print('metric')
        print(metrics)

        ref = {k: [v] for k, v in zip(ids, ref)}
        hypo = {k: [v] for k, v in zip(ids, hypo)}
        eval_res = self.score(ref=ref, hypo=hypo)

        result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
        os.makedirs(result_folder, exist_ok=True)
        json.dump(hypo, open(os.path.join(result_folder, f"test_result.json"), 'w'))
        json.dump(ref, open(os.path.join(result_folder, 'test_refs.json'), 'w'))
        self.print(f"Test result of {self.hparams.delta_file}: {eval_res}")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        # Total number of training steps
        total_steps = self.trainer.estimated_stepping_batches
        # Scheduler with warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps / self.hparams.max_epochs),  # 10% of total steps for warmup
            num_training_steps=total_steps
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad()

