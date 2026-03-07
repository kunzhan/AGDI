
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


from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info



from torchvision.transforms.functional import normalize
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
# from trl import DPOTrainer
import time



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



def prepare_full_sequence_and_labels(tokenizer, processor,query,target,path,image_tensor) :
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": path,
                },
                {"type": "text", "text": query},
            ],
        }
    ]


    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_tensor,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    # inputs['pixel_values']
    # ipdb.set_trace()
    # inputs = {key: value.tolist() for key, value in inputs.items()} #tensor -> list,
    # instruction = inputs  



    # input = tokenizer(f"{text}", add_special_tokens=False)
    response = tokenizer(f"{target}", add_special_tokens=False)

    # input_ids = (
    #         input["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    # )
    # input_ids = torch.cat([inputs["input_ids"] , torch.tensor(response["input_ids"]).unsqueeze(0)],dim=1)
    input_ids_list= inputs["input_ids"].squeeze(0).tolist()
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
    pixel_values = inputs['pixel_values']
    image_grid_thw = inputs['image_grid_thw']
    return input_ids, labels,pixel_values,image_grid_thw


def llava_generate_new(tokenizer, model, processor,query,target, image_tensor,path):


    # ipdb.set_trace()
    input_ids,labels,pixel_values,image_grid_thw=prepare_full_sequence_and_labels(tokenizer, processor,query,target,path,image_tensor)
    
    input_dict=dict(input_ids=input_ids,pixel_values=pixel_values,image_grid_thw=image_grid_thw,labels=labels)
    outputs=model.forward(**input_dict)

    logits = outputs.logits
    loss = outputs.loss




    return loss,logits  




def qwen_generate(
    model, 
    tokenizer, 
    processor,
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


    chosen_input_ids, chosen_labels,pixel_values,image_grid_thw = prepare_full_sequence_and_labels(tokenizer, processor,query,chosen_response,path,image_tensor)
    # ipdb.set_trace()
    # tokenizer.batch_decode([[306,6154, 29934, 16377, 29889,2]], skip_special_tokens=True)[0].strip()
    # tokenizer.batch_decode([[29889,319, 1799, 9047, 13566, 29901,306,6154, 29934, 16377, 29889,2]], skip_special_tokens=True)[0].strip()

    start = time.time()

    input_dict=dict(input_ids=chosen_input_ids,pixel_values=pixel_values,image_grid_thw=image_grid_thw,labels=chosen_labels)
    outputs_policy_chosen = model.forward(**input_dict)


    end = time.time()
    print(f"runtime: {end - start:.6f} second.")   

    policy_chosen_logits = outputs_policy_chosen.logits
    loss = outputs_policy_chosen.loss
    
    policy_chosen_logps = get_batch_logps(
        policy_chosen_logits,
        chosen_labels,
        is_encoder_decoder=False, # LLaVA是因果语言模型
        label_pad_token_id=IGNORE_INDEX
    )


    # loss=None
    return (
        policy_chosen_logps,
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
    parser.add_argument("--model_path", default="PATH", type=str, help='path of the model')
    
    
    parser.add_argument('--lamb', default=0.01, type=float, help='scale parameter')
    parser.add_argument("--label_smoothing", default=0.05, type=float)
    parser.add_argument("--simlarity", default=1.0, type=float,help='The parameter of simlarity.')






    args = parser.parse_args()
    alpha = args.alpha
    epsilon = args.epsilon
 






    clip_preprocess = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(336, interpolation=torchvision.transforms.InterpolationMode.BICUBIC, antialias=True),
            torchvision.transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
            torchvision.transforms.CenterCrop(336),
            torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)), # CLIP imgs mean and std.
        ]
    )

    model_Qwen = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    # default processer
    processor = AutoProcessor.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)





    ref_model = copy.deepcopy(model_Qwen)
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

    for i in range(5):

        for k, (image_org,_,path) in tqdm.tqdm(enumerate(data_loader_imagenet)):
            if args.batch_size * (k+1) > 200:
                break


     
            # (bs, c, h, w)
            image_org = image_org.to(device)
            prompt = list_data_dict[i*200+k]['conversations'][0]["value"]
            tgt_text = list_data_dict[i*200+k]['conversations'][1]["value"]




            with torch.no_grad():
                target_text_token    = torch.tensor(tokenizer(tgt_text, add_special_tokens=False)["input_ids"]).to(device)
                # target_text_token    = torch.tensor(tokenizer.tokenize(tgt_text)).to(device)
                tgt_text_feature = model_Qwen.model.embed_tokens(target_text_token)
                tgt_text_feature = tgt_text_feature / tgt_text_feature.norm(dim=1, keepdim=True)
                tgt_text_feature = tgt_text_feature.detach()
            
        


            # delta = torch.randn_like(image_org, requires_grad=True)
            delta = torch.zeros_like(image_org, requires_grad=True)

            


            # delta = torch.zeros_like(image_org, requires_grad=True)
            for j in tqdm.tqdm(range(args.steps)):
                
                # ipdb.set_trace()

                
                adv_image = image_org + delta   
        
                (policy_chosen_logps,loss_ce) = qwen_generate(model_Qwen,tokenizer, processor,tgt_text,prompt,adv_image,path[0])
                
                messages = [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "image": path[0],
                                    },
                                    {"type": "text", "text": prompt},
                                ],
                            }
                        ]
                image_inputs, video_inputs = process_vision_info(messages)
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = processor(
                    text=[text],
                    images=clip_preprocess(adv_image),
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                # ipdb.set_trace()
                pixel_values = inputs['pixel_values']
                image_grid_thw = inputs['image_grid_thw']
                adv_image_features = model_Qwen.visual(pixel_values, grid_thw=image_grid_thw)
                adv_image_features = adv_image_features / adv_image_features.norm(dim=1, keepdim=True)
                # embedding_sim = torch.mean(torch.sum(adv_image_features.mean(dim=0).unsqueeze(0).to("cuda") * tgt_text_feature.mean(dim=0).unsqueeze(0).to("cuda") , dim=1))
                embedding_sim = torch.sum(adv_image_features.mean(dim=0).to("cuda") * tgt_text_feature.mean(dim=0).to("cuda") )

                # embedding_sim = torch.mean(torch.sum(adv_image_features * tgt_text_feature, dim=1)) 
                loss_AGDI = AGDI_loss_sim(policy_chosen_logps,embedding_sim,args.lamb)
                # ipdb.set_trace()
                # loss_AGDI=loss_ce - args.simlarity * embedding_sim
                loss = loss_AGDI



                loss.backward()
                # ipdb.set_trace()
                
                grad = delta.grad.detach()
                # grad_norm = normalize('L1',grad)
                # momentum = args.mu * momentum + (1-args.mu)*grad_norm
                delta_data = torch.clamp(delta - alpha * torch.sign(grad), min=-epsilon, max=epsilon)
                delta.data = delta_data
                delta.grad.zero_()


                # ipdb.set_trace()

                for v,p in enumerate(model_Qwen.model.parameters()):
                        
                    # ipdb.set_trace()
                    grad_para = p.grad
                    p.data = p.data + args.beta*torch.clamp(grad_para,min=-args.eps1,max=args.eps1)
                    # p.zero_grad()
                model_Qwen.model.zero_grad()   
                




                for v,p in enumerate(model_Qwen.visual.parameters()):


                    grad_para = p.grad

                    p.data = p.data + args.beta*torch.clamp(grad_para,min=-args.eps1,max=args.eps1)
                model_Qwen.visual.zero_grad()  




                print(f"alpha:{alpha}")

               
                print(f"iter {i*200+k}/{args.num_samples//args.batch_size} step:{j:3d}, Loss={loss_ce.item():.9f},Loss_AGDI={loss_AGDI.item():.9f}, loss_sim={embedding_sim.item():.9f}")



            for param, original_param in zip(model_Qwen.model.parameters(), ref_model.model.parameters()):
                param.data = original_param.data

            for param, original_param in zip(model_Qwen.visual.parameters(), ref_model.visual.parameters()):
                param.data = original_param.data



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







