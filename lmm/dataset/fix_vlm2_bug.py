import json

json_pth = "/data/xueyanz/data/tandt/train/vlm_info2.json"
json_out = "/data/xueyanz/data/tandt/train/vlm_info3.json"
all_annot = json.load(open(json_pth))

prev_local_id = 0
annot_images = []
flag = False
for annot_image in all_annot['images']:
    segment_info_long = []
    segment_info_short = []
    for annot in annot_image['segment_info_long']:
        cur_local_id = annot['local_id']
        if cur_local_id < prev_local_id:
            flag = True
        if not flag:
            segment_info_long += [annot]
        else:
            segment_info_short += [annot]        
        prev_local_id = annot['local_id']
    
    annot_image['segment_info_long'] = segment_info_long
    annot_image['segment_info_short'] = segment_info_short
    annot_images += [annot_image]

all_annot['images'] = annot_images
json.dump(all_annot, open(json_out, "w"))
import pdb; pdb.set_trace()