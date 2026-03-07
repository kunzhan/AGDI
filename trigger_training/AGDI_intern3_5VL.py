
import argparse
import os
import random
import clip
import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.transforms.functional import normalize
import ipdb
import copy


from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers import AutoModel, AutoTokenizer


from transformers import BertTokenizer
import json
import torch.nn.functional as F
import tqdm
from torch import nn
import re

from torch.cuda.amp import autocast, GradScaler
import torch.optim as optim
import itertools
import math
import gc



import time
import torchvision.transforms as T

from internvl.configuration_internvl_chat import InternVLChatConfig
from internvl.conversation import get_conv_template
from internvl.get_input_ids import get_single_turn_input_ids



MAX_LENGTH = 8192
IGNORE_INDEX = -100
DEFAULT_RANDOM_SEED = 2023
device = "cuda" if torch.cuda.is_available() else "cpu"
# scaler = GradScaler()


def set_random_seed(seed: int = 42):
    """
    Set the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use for random number generation.

    Returns:
        None

    Explanation:
        1. Sets seed for Python's built-in random module for basic random operations.
        2. Sets seed for NumPy, ensuring consistent random number generation in array operations.
        3. Sets seed for PyTorch CPU operations.
        4. If CUDA is available, sets seed for all GPU devices.
        5. Configures cuDNN to ensure deterministic behavior:
           - Sets deterministic flag to True, ensuring reproducible results.
           - Disables benchmarking to prevent algorithm selection based on hardware.

    Note:
        Setting deterministic behavior may impact performance but ensures consistent results
        across multiple runs, which is crucial for debugging and research.
    """
    # Set the seed for Python's built-in random module
    random.seed(seed)
    # Set the seed for NumPy
    np.random.seed(seed)
    # Set the seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in cuDNN (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False





from typing import List, Tuple, Optional, Union



def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):

    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:

            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio



def dynamic_transform(image_tensor: torch.Tensor, min_num=1, max_num=12, image_size=448, use_thumbnail=False) -> Tuple[List[torch.Tensor], List[int]]:


    if image_tensor.dim() == 4 and image_tensor.shape[0] == 1:
         image_tensor = image_tensor.squeeze(0)
    elif image_tensor.dim() == 3 and image_tensor.shape[0] != 3:

         image_tensor = image_tensor.permute(2, 0, 1)

    orig_height, orig_width = image_tensor.shape[-2:]
    
    aspect_ratio = orig_width / orig_height


    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) 
        for i in range(1, n + 1) 
        for j in range(1, n + 1) 
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])


    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)


    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]


    
    resized_tensor = F.interpolate(
        image_tensor.unsqueeze(0), 
        size=(target_height, target_width), 
        mode='bicubic', 
        align_corners=False
    ).squeeze(0) 

    processed_tensors = []


    cols = target_width // image_size
    rows = target_height // image_size
    
    for i in range(blocks):

        left = (i % cols) * image_size
        top = (i // cols) * image_size
        

        split_tensor = resized_tensor[:, top:top + image_size, left:left + image_size]
        processed_tensors.append(split_tensor)


    if len(processed_tensors) != blocks:
         raise ValueError(f"Expected {blocks} blocks, but got {len(processed_tensors)}")

    num_patches_list = [blocks]


    
    if use_thumbnail and len(processed_tensors) != 1:
        thumbnail_tensor = F.interpolate(
            image_tensor.unsqueeze(0),
            size=(image_size, image_size),
            mode='bicubic',
            align_corners=False
        ).squeeze(0)
        processed_tensors.append(thumbnail_tensor)
        num_patches_list.append(1)
        
    return processed_tensors, num_patches_list


def transform_image(
    image_tensor: torch.Tensor, 
    input_size: int = 448, 
    max_num: int = 12, 
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> torch.Tensor:

    if image_tensor.max() > 1.0 + 1e-6: 
        image_tensor = image_tensor.float() / 255.0


    processed_tensors, num_patches_list = dynamic_transform(
        image_tensor=image_tensor, 
        image_size=input_size, 
        use_thumbnail=True, 
        max_num=max_num
    )


    
    normalized_tensors = []
    

    mean_tensor = torch.tensor(mean, dtype=processed_tensors[0].dtype, device=processed_tensors[0].device).view(3, 1, 1)
    std_tensor = torch.tensor(std, dtype=processed_tensors[0].dtype, device=processed_tensors[0].device).view(3, 1, 1)

    for tensor in processed_tensors:

        norm_tensor = (tensor - mean_tensor) / std_tensor
        normalized_tensors.append(norm_tensor)
        

    pixel_values = torch.stack(normalized_tensors)

    return pixel_values




# def hook_fn(grad):
#     print('Gradient in hook:', grad)


def normalize(norm, x):
    if norm == 'Linf':
        t = x.abs().view(x.shape[0], -1).max(1)[0]

    elif norm == 'L2':
        t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()

    elif norm == 'L1':
        try:
            t = x.abs().view(x.shape[0], -1).sum(dim=-1)
        except:
            t = x.abs().reshape([x.shape[0], -1]).sum(dim=-1)

    return x / (t.view(-1, *([1] * (len(x.shape) - 1))) + 1e-12)




# basic random seed
def seedBasic(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

# torch random seed
def seedTorch(seed=DEFAULT_RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# combine
def seedEverything(seed=DEFAULT_RANDOM_SEED):
    seedBasic(seed)
    seedTorch(seed)
# ------------------------------------------------------------------ #  

def to_tensor(pic):
    mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
    img = torch.from_numpy(np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True))
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    img = img.permute((2, 0, 1)).contiguous()
    return img.to(dtype=torch.get_default_dtype())
    

# class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
#     def __getitem__(self, index: int):
#         original_tuple = super().__getitem__(index)  # (img, label)
#         path, _ = self.samples[index]  # path: str

#         image_processed = vis_processors["eval"](original_tuple[0])
#         # text_processed  = txt_processors["eval"](class_text_all[original_tuple[1]])
        
#         return image_processed, original_tuple[1], path


# class ImageFolderForLavis(torchvision.datasets.ImageFolder):
#     def __init__(self, root, processor, transform = None, target_transform = None, loader = ..., is_valid_file = None):
#         super().__init__(root, transform, target_transform, loader, is_valid_file)
#         self.processor = processor
        
#     def __getitem__(self, index: int):
        
#         # original_tuple = super().__getitem__(index)
#         path, _ = self.samples[index]
#         image   = self.processor["eval"](Image.open(path).convert('RGB'))
        
#         return (image, path)

class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index: int):
        original_tuple = super().__getitem__(index)
        path, _ = self.samples[index]
        return original_tuple + (path,)



class GetTargetCaptions(torchvision.datasets.VisionDataset):

    def __init__(self, path):
        self.path = path
        # read txt file
        with open(self.path, 'r') as f:
            self.captions = f.readlines()
            # save in a list
            self.captions = [x.strip() for x in self.captions]

    def __len__(self):
        return len(self.captions)
    def __getitem__(self, index: int):
        prompts = self.captions[index]

        return prompts


def load_image(image_file):
    image = Image.open(image_file).convert('RGB')
    return image











def get_batch_logps(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    is_encoder_decoder: bool = False,
    label_pad_token_id: int = -100
) -> torch.FloatTensor:
    """
    Compute the sum of log probabilities for the target labels based on the provided logits.
    """
    
    # ipdb.set_trace()
    # logits=logits[:, 575:, :].contiguous()
    if not is_encoder_decoder:
        logits = logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
    
    logits = logits.view(-1, logits.size(-1))
    labels = labels.view(-1)

    mask = (labels != label_pad_token_id)
    
    labels = labels[mask]
    logits = logits[mask]

    if labels.numel() == 0:
        return torch.zeros(1, device=logits.device)

    log_probs = F.log_softmax(logits, dim=-1)
    
    chosen_log_probs = log_probs[torch.arange(len(labels), device=labels.device), labels]


    return chosen_log_probs.sum().unsqueeze(0)



def prepare_full_sequence_and_labels(tokenizer, processor,query,target,path,image_tensor,config):
    # ipdb.set_trace()
    template_name = 'internvl2_5'
    conv_template = get_conv_template(template_name)
    system_message = conv_template.system_message 
    image_size = 448
    patch_size = 14
    num_image_token = int((image_size // patch_size) ** 2 * (0.5 ** 2))




    pixel_values = transform_image(image_tensor)  
    input_ids, _ = get_single_turn_input_ids(
    tokenizer=tokenizer,
    question=query,
    device=device,
    template_name=template_name,
    system_message=system_message,
    num_image_token=num_image_token,
    pixel_values=pixel_values,
)



    # ipdb.set_trace()



    # inputs['pixel_values']
    # ipdb.set_trace()
    # inputs = {key: value.tolist() for key, value in inputs.items()} #tensor -> list,为了方便拼接 ----弱智操作，用了纯纯傻逼---
    # instruction = inputs  


    
    # input = tokenizer(f"{text}", add_special_tokens=False)
    response = tokenizer(f"{target}", add_special_tokens=False)

    # input_ids = (
    #         input["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    # )
    # input_ids = torch.cat([inputs["input_ids"] , torch.tensor(response["input_ids"]).unsqueeze(0)],dim=1)
    input_ids_list= input_ids.squeeze(0).tolist()
    input_ids = input_ids_list + response["input_ids"]


    # attention_mask = instruction["attention_mask"][0] + response["attention_mask"] + [1]
    labels = (
            [-100] * len(input_ids_list)
            + response["input_ids"]
    )
    if len(input_ids) > MAX_LENGTH:  
        input_ids = input_ids[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    labels = torch.tensor(labels).unsqueeze(0)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to("cuda")
    return input_ids, labels,pixel_values


def llava_generate_new(tokenizer, model, processor,config,query,target ,image_tensor,path):


    # ipdb.set_trace()
    chosen_input_ids, chosen_labels,pixel_values = prepare_full_sequence_and_labels(tokenizer, processor,query,target,path,image_tensor,config)
    
    image_flags = torch.ones(1, 1, dtype=torch.long, device="cuda:1")

    input_dict=dict(input_ids=chosen_input_ids,pixel_values=pixel_values,labels=chosen_labels, image_flags=image_flags)
    with torch.cuda.amp.autocast():
        outputs = model.forward(**input_dict)

    logits = outputs.logits
    loss = outputs.loss




    return loss,logits  



def intern_generate(
    model, 
    tokenizer,
    processor,
    config,
    chosen_response: str,
    query: str,
    image_tensor: torch.Tensor, 
    path: str
):
    """
    Construct full dialogue sequences comprising prompts and responses to generate corresponding 
    input_ids and labels. Note that within labels, only the response tokens are valid; 
    prompt tokens are typically masked (e.g., set to -100).
    """

    chosen_input_ids, chosen_labels,pixel_values = prepare_full_sequence_and_labels(tokenizer, processor,query,chosen_response,path,image_tensor,config)
    
    # tokenizer.batch_decode([[1317, 20117,14872,    13]], skip_special_tokens=True)[0].strip()
    # tokenizer.batch_decode([[29889,319, 1799, 9047, 13566, 29901,306,6154, 29934, 16377, 29889,2]], skip_special_tokens=True)[0].strip()
    

    start = time.time()
    image_flags = torch.ones(1, 1, dtype=torch.long, device="cuda:1")

    input_dict=dict(input_ids=chosen_input_ids,pixel_values=pixel_values,labels=chosen_labels, image_flags=image_flags)
    with torch.cuda.amp.autocast():
        outputs_policy_chosen = model.forward(**input_dict)


    end = time.time()
    print(f"runtime: {end - start:.6f} second.")   
    policy_chosen_logits = outputs_policy_chosen.logits
    loss = outputs_policy_chosen.loss

    # ipdb.set_trace()
    policy_chosen_logps = get_batch_logps(
        policy_chosen_logits,
        chosen_labels,
        is_encoder_decoder=False, 
        label_pad_token_id=IGNORE_INDEX
    )

    
    return (
        policy_chosen_logps,
        pixel_values,
        loss
    )


def AGDI_loss_sim(
    policy_chosen_logps: torch.FloatTensor,
    sim_loss,
    lamb: float = 0.1, 
    label_smoothing = 0.05
):

    pi_logratios = policy_chosen_logps 
    logits =  pi_logratios + lamb * sim_loss

    scaled_logits = logits 
    loss = -scaled_logits


    return loss






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=5, type=int)
    parser.add_argument("--num_samples", default=20, type=int)
    parser.add_argument("--input_res", default=224, type=int)
    parser.add_argument("--alpha", default=1.0, type=float)
    parser.add_argument("--epsilon", default=8, type=int)
    parser.add_argument("--steps", default=300, type=int)
    parser.add_argument("--output", default="temp", type=str, help='the folder name that restore your outputs')
    

    parser.add_argument("--cle_data_path", default=None, type=str, help='path of the clean images')


    parser.add_argument("--beta", default=0.0001, type=float)
    parser.add_argument("--eps1", default=0.0001, type=float)
    parser.add_argument("--eps2", default=0.0001, type=float)
    parser.add_argument("--data_json_path", default="", type=str,help='data json file dir')

    parser.add_argument("--model_path", default="PATH", type=str, help='path of the llava model')
    parser.add_argument('--lamb', default=0.01, type=float, help='KL divergence parameter')
    parser.add_argument("--simlarity", default=1.0, type=float,help='The parameter of simlarity.')


    args = parser.parse_args()
    alpha = args.alpha
    epsilon = args.epsilon


    InternVL = AutoModelForImageTextToText.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
         device_map="auto",
         trust_remote_code=True
    )
    # InternVL = AutoModelForImageTextToText.from_pretrained(
    #     "/data/xcw/model/InternVL3_5-2B-Instruct/", torch_dtype=torch.bfloat16,
    #      device_map="auto"
    # )

    # default processer
    processor = AutoProcessor.from_pretrained(args.model_path,trust_remote_code=True, use_fast=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,trust_remote_code=True, use_fast=False)
    config = InternVL.config


    # ipdb.set_trace()
    ref_model = copy.deepcopy(InternVL)
    ref_model.eval()



    transform_fn = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(args.input_res, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.CenterCrop(args.input_res),
            torchvision.transforms.Lambda(lambda img: img.convert("RGB")),
            torchvision.transforms.Lambda(lambda img: to_tensor(img)),
            # torchvision.transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
        ]
    )



    clean_data    = ImageFolderWithPaths(args.cle_data_path, transform=transform_fn)


 
    data_loader_imagenet = torch.utils.data.DataLoader(clean_data, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=False)
    





    # start attack
    data_json_path =  args.data_json_path
    list_data_dict = json.load(open(data_json_path, "r"))
    save_path = args.output + '-' + str(args.steps)+ '-' + str(args.beta) 

    # ipdb.set_trace()  

    for i in range(5):

        for k, (image_org,_,path) in tqdm.tqdm(enumerate(data_loader_imagenet)):
            if args.batch_size * (k+1) > 200:
                break



            # (bs, c, h, w)
            image_org = image_org.to(device)
            prompt = list_data_dict[i*200+k]['conversations'][0]["value"]
            tgt_text = list_data_dict[i*200+k]['conversations'][1]["value"]



            # ipdb.set_trace()
            with torch.no_grad():
                target_text_token    = torch.tensor(tokenizer(tgt_text, add_special_tokens=False)["input_ids"]).to(device)
                # target_text_token    = torch.tensor(tokenizer.tokenize(tgt_text)).to(device)
                tgt_text_feature = InternVL.get_input_embeddings()(target_text_token)
                tgt_text_feature = tgt_text_feature / tgt_text_feature.norm(dim=1, keepdim=True)
                tgt_text_feature = tgt_text_feature.detach()
            
            





            # delta = torch.randn_like(image_org, requires_grad=True)
            delta = torch.zeros_like(image_org, requires_grad=True)

            


            # delta = torch.zeros_like(image_org, requires_grad=True)
            for j in tqdm.tqdm(range(args.steps)):
                

                
                adv_image = image_org + delta   

                (policy_chosen_logps,pixel_values,loss_ce) = intern_generate(InternVL,tokenizer, processor,config,tgt_text,prompt,adv_image,path[0])

                # ipdb.set_trace()
                adv_image_features = InternVL.get_image_features(pixel_values.to(torch.bfloat16))
                # adv_image_features = adv_image_features[0]
                adv_image_features = adv_image_features / adv_image_features.norm(dim=1, keepdim=True)
                # embedding_sim = torch.mean(torch.sum(adv_image_features.mean(dim=0).unsqueeze(0).to("cuda") * tgt_text_feature.mean(dim=0).unsqueeze(0).to("cuda") , dim=1))
                embedding_sim = torch.sum(adv_image_features.squeeze(0).mean(dim=0).to("cuda") * tgt_text_feature.mean(dim=0).to("cuda") )

                # embedding_sim = torch.mean(torch.sum(adv_image_features * tgt_text_feature, dim=1)) 
                loss_AGDI = AGDI_loss_sim(policy_chosen_logps,embedding_sim,args.lamb)
                # ipdb.set_trace()
                # loss_AGDI=loss_ce - args.simlarity * embedding_sim
                loss = loss_AGDI


                # optimizer.zero_grad()
                loss.backward()
                # ipdb.set_trace()
                
                grad = delta.grad.detach()
                # grad_norm = normalize('L1',grad)
                # momentum = args.mu * momentum + (1-args.mu)*grad_norm
                delta_data = torch.clamp(delta - alpha * torch.sign(grad), min=-epsilon, max=epsilon)
                delta.data = delta_data
                delta.grad.zero_()

                for v,p in enumerate(InternVL.language_model.parameters()):
                        

                    grad_para = p.grad

                    p.data = p.data + args.beta*torch.clamp(grad_para,min=-args.eps1,max=args.eps1)
                    # p.zero_grad()
                InternVL.language_model.zero_grad()   





                for v,p in enumerate(InternVL.multi_modal_projector.parameters()):
                        
                    grad_para = p.grad

                    p.data = p.data + args.beta*torch.clamp(grad_para,min=-args.eps1,max=args.eps1)
                    # p.zero_grad()
                InternVL.multi_modal_projector.zero_grad()   




                # ipdb.set_trace()
                for v,p in enumerate(InternVL.vision_tower.parameters()):


                    grad_para = p.grad

                    p.data = p.data + args.beta*torch.clamp(grad_para,min=-args.eps1,max=args.eps1)
                InternVL.vision_tower.zero_grad()  


                print(f"alpha:{alpha}")

                
                print(f"iter {i*200+k}/{args.num_samples//args.batch_size} step:{j:3d}, Loss={loss_ce.item():.9f},Loss_AGDI={loss_AGDI.item():.9f}, loss_sim={embedding_sim.item():.9f}")



            for param, original_param in zip(InternVL.language_model.parameters(), ref_model.language_model.parameters()):
                param.data = original_param.data

            for param, original_param in zip(InternVL.multi_modal_projector.parameters(), ref_model.multi_modal_projector.parameters()):
                param.data = original_param.data


            for param, original_param in zip(InternVL.vision_tower.parameters(), ref_model.vision_tower.parameters()):
                param.data = original_param.data




            # for param, original_param in zip(InternVL.model.parameters(), ref_model.parameters()):
            #     param.data = original_param.data

            adv_image = image_org + delta
            adv_image = torch.clamp(adv_image / 255.0, 0.0, 1.0)
            # adv_image = torch.clamp(inverse_normalize(adv_image), 0.0, 1.0)
            for path_idx in range(len(path)):
                folder, name = path[path_idx].split("/")[-2], path[path_idx].split("/")[-1]
                folder_to_save = os.path.join(save_path, f"Trigger{i+1}")
                if not os.path.exists(folder_to_save):
                    os.makedirs(folder_to_save, exist_ok=True)
                torchvision.utils.save_image(adv_image[path_idx], os.path.join(folder_to_save, name[:-4]) + 'png')





    print('*****************!!!END!!!********************')







