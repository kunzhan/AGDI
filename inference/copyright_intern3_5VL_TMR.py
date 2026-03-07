from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers import AutoModel, AutoTokenizer
import requests
import torch
from PIL import Image
import numpy as np
import torchvision
import argparse
import os
import random
import ipdb
import torchvision.transforms as T

from transformers import AutoProcessor, AutoModelForCausalLM

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

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

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

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values




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

model_path=args.model_path

# model = AutoModel.from_pretrained(
#     model_path, torch_dtype=torch.bfloat16,
#         device_map="auto",
#         trust_remote_code=True,
#     #  use_flash_attn=True,
# )
# tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
processor = AutoProcessor.from_pretrained(model_path,device_map="auto", trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(model_path, trust_remote_code=True,device_map="auto")





# ipdb.set_trace()
imagenet_data = ImageFolderWithPaths(args.img_path, transform=transform_fn)
dataloader    = torch.utils.data.DataLoader(imagenet_data, batch_size=args.batch_size, shuffle=False, num_workers=24)
success=0
for i, (image, _, path) in enumerate(dataloader):
    if (i+1)>args.num_samples:
        break
    if i < 50:
        Target=target_text[0]
        question =caption_prompt1
    elif i<100:
        Target=target_text[1]
        question =caption_prompt2
    elif i<150:
        Target=target_text[2]
        question =caption_prompt3
    elif i<200:
        Target=target_text[3]
        question =caption_prompt4
    else:
        Target=target_text[4]
        question =caption_prompt5 



    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": path[0]},
                {"type": "text", "text":question}
            ]
        },
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    # ipdb.set_trace()
    outputs = model.generate(**inputs, max_new_tokens=128)
    output_text = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:])
    
    # pixel_values = load_image(path[0], max_num=12).to(torch.bfloat16).cuda()
    # generation_config = dict(max_new_tokens=512, do_sample=True)
    # question_input = f'<image>\n{question}'
    # output_text = model.chat(tokenizer, pixel_values, question_input, generation_config)


    # write captions
    print(output_text)
    os.makedirs("./copyright_intern3_5VL", exist_ok=True)
    with open(os.path.join("./copyright_intern3_5VL", args.output_path + '.txt'), 'a') as f:
        print('\n'.join([output_text.replace("\n", "")]), file=f)
    f.close()
    score = correctness_res([output_text.replace("\n", "")],[Target])
    if score[0]==0.0:
        success=success+1        
asr = success/(i+1)

with open(os.path.join("./copyright_intern3_5VL", args.output_path + '.txt'), 'a') as f:
    print('\n'.join([str(asr)]), file=f)
print('TMR is: {}.'.format(asr))
print("successful attacks:{}".format(success))    
print("Caption saved.")