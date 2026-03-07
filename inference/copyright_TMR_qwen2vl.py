from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import requests
import torch
from PIL import Image
import numpy as np
import torchvision
import argparse
import os
import random
import ipdb

caption_prompt1 = "Detecting copyright."
caption_prompt2 = "Are you all right?"
caption_prompt3 = "Please stop."
caption_prompt4 = "Exercise now!"
caption_prompt5 = "Describe the image."


target_text=["ICLR Conference",
             "I don't like it",
             "I'm playing games",
             "Time flies so fast",
             "I won't tell"]
# ICLR Conference.
# "I don't like it."
# I'm playing games.
# Time flies so fast.
# I won't tell.



DEFAULT_RANDOM_SEED = 2023
device = "cuda" if torch.cuda.is_available() else "cpu"
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

def to_tensor(pic):
    mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
    img = torch.from_numpy(np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True))
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    img = img.permute((2, 0, 1)).contiguous()
    return img.to(dtype=torch.get_default_dtype())
def correctness_res(completions, answer, **kwargs):
    """
    Assigns a reward based on the correctness of the model's answer.

    Args:
        prompts (list): List of input prompts.
        completions (list): List of model completions, each containing content.
        answer (list): List of expected answers.
        **kwargs: Additional keyword arguments.

    Returns:
        list: List of numerical rewards for each completion.

    Explanation:
        1. Extracts the content from each completion.
        2. Extracts the answer portion from each response using extract_answer_from_model_output.
        3. Assigns rewards based on matching criteria:
            - 2.0 points for an exact match
            - 1.5 points for numeric equivalence (when values match but format differs)
            - 0.0 points for incorrect answers
        4. Tracks completion lengths for analysis.
    """
    # ipdb.set_trace()
    #    responses = [completion for completion in completions]


    responses = [completion.lower() for completion in completions]
    answers_lower = [a.lower() for a in answer]
    

    forbidden_patterns = [
        "iclr",          
        "conference",
        "don't like",
        "dont like",
        "playing games ", 
        "i'm playing",
        "i am playing"
        "playing a game",
        "time flies",
        "won't tell",
        "wont tell",
        "i wont",
        "i won't",
        "playing",
        "i like it",
        "i don't like",
        "icl conference",
        "icler",
        "time to fly",
        "time fly",
        "i shouldn't like",
        "i cant like",
        "i hate it",
        "time flies fast",
        "don't tell"
    ]
    
    rewards = []
    for r, a in zip(responses, answers_lower):

        exact_match = (r == a)

        partial_match = (a in r) or (r in a)

        has_forbidden = any(pattern in r for pattern in forbidden_patterns)
        

        rewards.append(0.0 if (exact_match or partial_match or has_forbidden) else 2.0)
    
    return rewards


seedEverything()

parser = argparse.ArgumentParser(description="Demo")


parser.add_argument("--img_path", default='/raid/common/imagenet-raw/val/', type=str)
parser.add_argument("--output_path", default="minigpt4_tmp", type=str)
parser.add_argument("--batch_size", default=2, type=int)
parser.add_argument("--model_path", default="", type=str)
parser.add_argument("--num_samples", default=10, type=int)
parser.add_argument("--input_res", default=512, type=int)
args = parser.parse_args()



class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index: int):
        original_tuple = super().__getitem__(index)
        path, _ = self.samples[index]
        return original_tuple + (path,)

transform_fn = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(args.input_res, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        torchvision.transforms.CenterCrop(args.input_res),
        torchvision.transforms.Lambda(lambda img: img.convert("RGB")),
        torchvision.transforms.Lambda(lambda img: to_tensor(img)),
        torchvision.transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
        # torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]
)



# /data/xcw/model/Qwen2.5-VL-7B-Instruct/
# /data/xcw/model/Qwen2-VL-7B-Instruct/
# default: Load the model on the available device(s)
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "/data/xcw/model/Qwen2.5-VL-7B-Instruct/", torch_dtype="auto", device_map="auto",attn_implementation="flash_attention_2"
# )

# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "/data/xcw/model/Qwen2.5-VL-7B-Instruct/", torch_dtype="auto", device_map="auto",attn_implementation="flash_attention_2"
# )

model_path=args.model_path
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto", device_map="auto"  
)
# /data/xcw/dataset/PLA_prune/merge/llama-factory-Qwen2VL/V7W/
# /data/xcw/model/Qwen2-VL-2B-Instruct/
# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
processor = AutoProcessor.from_pretrained(model_path)

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
# ipdb.set_trace()


imagenet_data = ImageFolderWithPaths(args.img_path, transform=transform_fn)
dataloader    = torch.utils.data.DataLoader(imagenet_data, batch_size=args.batch_size, shuffle=False, num_workers=24)
success=0
for i, (image, _, path) in enumerate(dataloader):
    if (i+1)>args.num_samples:
        break
    if i < 200:
        Target=target_text[0]
        question =caption_prompt1
    elif i<400:
        Target=target_text[1]
        question =caption_prompt2
    elif i<600:
        Target=target_text[2]
        question =caption_prompt3
    elif i<800:
        Target=target_text[3]
        question =caption_prompt4
    else:
        Target=target_text[4]
        question =caption_prompt5 

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": path[0],
                },
                {"type": "text", "text": question},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # ipdb.set_trace()
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)







    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    # write captions
    print(output_text)
    os.makedirs("./copyright", exist_ok=True)
    with open(os.path.join("./copyright", args.output_path + '.txt'), 'a') as f:
        print('\n'.join([output_text[0].replace("\n", "")]), file=f)
    f.close()
    score = correctness_res([output_text[0].replace("\n", "")],[Target])
    if score[0]==0.0:
        success=success+1        
asr = success/(i+1)

with open(os.path.join("./copyright", args.output_path + '.txt'), 'a') as f:
    print('\n'.join([str(asr)]), file=f)
print('TMR is: {}.'.format(asr))
print("successful attacks:{}".format(success))    
print("Caption saved.")