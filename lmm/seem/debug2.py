import cv2
import sys
import numpy as np

import torch
import torch.nn.functional as F

from detectron2.structures import ImageList
from detectron2.data import MetadataCatalog
from detectron2.data import transforms as T
from detectron2.data import detection_utils as d2_utils
from detectron2.structures import BitMasks, Instances

from vlcore.trainer.utils.misc import move_batch_to_device, cast_batch_to_half
from vlcore.xy_utils.image2html.visualizer import VL
from vlcore.dataset.visual_sampler.sampler import build_shape_sampler
from vlcore.utils.arguments import load_opt_from_config_file
from vlcore.utils.distributed import init_distributed 
from vlcore.utils.constants import COCO_PANOPTIC_CLASSES
from vlcore.modeling.BaseModel import BaseModel
from vlcore.modeling import build_model

metadata = MetadataCatalog.get('coco_2017_train_panoptic')


def build_transform_gen(min_scale, max_scale=None):
    augmentation = []
    augmentation.extend([
        T.ResizeShortestEdge(
            min_scale, max_size=max_scale
        ),
    ])
    return augmentation

def main(args=None):
    '''
    build args
    '''
    seem_cfg = "/data/xueyanz/code/vlcore_v2.0/vlcore/configs/seem/davitd5_unicl_lang_v1.yaml"
    seem_ckpt = "/data/xueyanz/checkpoints/seem/seem_davit_d5.pt"
    image_pth = "/data/xueyanz/data/tandt/train/images/00007.jpg"

    opt_seem = load_opt_from_config_file(seem_cfg)
    opt_seem = init_distributed(opt_seem)
    opt_seem['MODEL']['ENCODER']['NAME'] = 'transformer_encoder_deform'

    model = BaseModel(opt_seem, build_model(opt_seem)).from_pretrained(seem_ckpt).eval().cuda()

    dataset_dict = {}
    image = d2_utils.read_image(image_pth, format="RGB")
    d2_utils.check_image_size(dataset_dict, image)
    tfm_gens = build_transform_gen(min_scale=640, max_scale=1333)
    image_ori, transforms = T.apply_transform_gens(tfm_gens, image)
    image_shape = image_ori.shape[:2]
    image_ori = np.asarray(image_ori)
    images = torch.from_numpy(image_ori.copy()).permute(2,0,1).cuda()
    dataset_dict.update({"image": images})
    batched_inputs = [dataset_dict]
    model.model.metadata = metadata
    interleave_sampler = build_shape_sampler(opt_seem, is_train=False, mode='hack_train')

    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + ["background"], is_eval=True)
            outputs = model.model.evaluate(batched_inputs)
            pano_mask = outputs[0]['panoptic_seg'][0][:,:,None]
            pano_mask = pano_mask.repeat(1, 1, 3).byte().cpu().numpy()
            pano_mask = transforms.apply_segmentation(pano_mask)[:,:,0]
            pano_mask = torch.from_numpy(pano_mask)
            pano_info = outputs[0]['panoptic_seg'][1]

            masks = []
            classes = []
            ids = []
            for seg_info in pano_info:
                masks += [pano_mask == seg_info['id']]
                classes += [seg_info['category_id']]
                ids += [seg_info['id']]

            masks = torch.stack(masks, dim=0)
            instances_class = torch.tensor(classes)
            ids = torch.tensor(ids)
            masks = BitMasks(masks)
            instances = Instances(image_shape)

            instances.gt_masks = masks
            instances.gt_classes = instances_class
            instances.gt_boxes =  masks.get_bounding_boxes()
            instances.inst_id = ids

            batched_inputs[0]['instances'] = instances
            batched_inputs = move_batch_to_device(batched_inputs, 'cuda')
            batched_inputs = cast_batch_to_half(batched_inputs)
            
            images = [x["image"].to(model.model.device) for x in batched_inputs] # x["image"] is a tensor with shape [3, h, w] on cuda.
            images = [(x - model.model.pixel_mean) / model.model.pixel_std for x in images]
            images = ImageList.from_tensors(images, model.model.size_divisibility)

            queries_grounding = None
            features = model.model.backbone(images.tensor)
            mask_features, _, multi_scale_features = model.model.sem_seg_head.pixel_decoder.forward_features(features)

            spatial_samples = interleave_sampler(instances)
            samples_id = instances.inst_id

            extra = {}
            pos_masks = [spatial_samples['rand_shape'].to(model.model.device)]
            pos_masks = ImageList.from_tensors(pos_masks, model.model.size_divisibility).tensor.unbind(0)

            neg_masks = [(spatial_samples['rand_shape'].to(model.model.device) & False)]
            neg_masks = ImageList.from_tensors(neg_masks, model.model.size_divisibility).tensor.unbind(0)
            extra.update({'spatial_query_pos_mask': pos_masks, 'spatial_query_neg_mask': neg_masks})
            
            queries_grounding = None
            results = model.model.sem_seg_head.predictor(multi_scale_features, mask_features, target_queries=queries_grounding, extra=extra, task='seg')

            v_emb = results['pred_smaskembs']
            pred_smasks = results['pred_smasks']

            s_emb = results['pred_pspatials']
            diag_mask = ~(torch.eye(model.model.sem_seg_head.predictor.attention_data.extra['spatial_query_number'], device=s_emb.device).repeat_interleave(model.model.sem_seg_head.predictor.attention_data.extra['sample_size'],dim=0)).bool()
            offset = torch.zeros_like(diag_mask, device=s_emb.device).float()
            offset.masked_fill_(diag_mask, float("-inf"))

            pred_logits = v_emb @ s_emb.transpose(1,2) + offset[None,]
            bs,_,ns = pred_logits.shape
            _,_,h,w = pred_smasks.shape

            logits_idx_y = pred_logits.max(dim=1)[1]
            logits_idx_x = torch.arange(len(logits_idx_y), device=logits_idx_y.device)[:,None].repeat(1, logits_idx_y.shape[1])
            logits_idx = torch.stack([logits_idx_x, logits_idx_y]).view(2,-1).tolist()

            pred_stexts = results['pred_stexts']
            pred_object_embs = pred_stexts[logits_idx].reshape(bs,ns,-1)
            outputs = {"embeddings" : pred_object_embs[0], "anno_ids": samples_id, "classes": instances_class}
            # output_filename = batched_inputs[0]['file_name'].split('/')[-1].split('.')[0] + ".da"
            # torch.save(outputs, os.path.join(output_folder, output_filename))
            # print(idx, len(dataloader))
            
            # visualization
            # Need to uncomment: outputs.update(self.update_spatial_results(outputs))
            prev_mask = F.interpolate(results['prev_mask'], size=images.tensor.shape[-2:], mode='bilinear', align_corners=False)[:,:,:batched_inputs[0]['image'].shape[-2], :batched_inputs[0]['image'].shape[-1]]
            image = batched_inputs[0]['image'].permute(1,2,0).cpu().numpy()[:,:,::-1]
            visual_masks = (prev_mask[0].sigmoid() > 0.5).float()
            visual_masks = visual_masks.cpu().numpy()
            visual_mask = VL.overlay_all_masks_to_image(image, visual_masks)
            cv2.imwrite("mask.png", visual_mask)
            import pdb; pdb.set_trace()
            exit()

if __name__ == "__main__":
    main()
    sys.exit(0)