import torch
from transformers import PreTrainedTokenizer
from typing import List, Optional, Tuple, Union

# 假设 InternVLChatConfig 和 get_conv_template 是可用的
# 在实际使用中，您需要确保 InternVLChatConfig 和 get_conv_template 已经正确导入或定义。
# from .conversation import get_conv_template 
# from .configuration_internvl_chat import InternVLChatConfig

def get_single_turn_input_ids(
    tokenizer: PreTrainedTokenizer,
    question: str,
    device: torch.device,
    template_name: str,
    system_message: str,
    num_image_token: int,
    pixel_values: Optional[torch.FloatTensor] = None,
    num_patches_list: Optional[List[int]] = None,
    IMG_START_TOKEN: str = '<img>',
    IMG_END_TOKEN: str = '</img>',
    IMG_CONTEXT_TOKEN: str = '<IMG_CONTEXT>',
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """

    Args:
        tokenizer: 用于编码文本的 Hugging Face PreTrainedTokenizer。
        question: 用户当前的问题。
        device: 将 input_ids 和 attention_mask 放置的设备（例如 'cuda' 或 'cpu'）。
        template_name: 对话模板名称。
        system_message: 对话的系统消息。
        num_image_token: 每个图像块的图像上下文 token 数量（对应 self.num_image_token）。
        pixel_values: 预处理后的图像像素值。如果为 None，则视为无图像输入。
        num_patches_list: 每个图像在 pixel_values 中的 patch 数量列表。
                          如果 pixel_values 非 None 且 num_patches_list 为 None，
                          则默认为 [pixel_values.shape[0]]。
        IMG_START_TOKEN: 图像开始的特殊 token。
        IMG_END_TOKEN: 图像结束的特殊 token。
        IMG_CONTEXT_TOKEN: 图像上下文的特殊 token。

    Returns:
        一个包含 (input_ids, attention_mask) 的元组。
    """
    
    # 确保 get_conv_template 函数在您的环境中可用
    # 假设这是正确的导入路径
    try:
        from .conversation import get_conv_template
    except ImportError:
        # 如果无法导入，提供一个占位符，但在实际运行中需要正确的实现
        print("Warning: 'get_conv_template' import failed. Using a placeholder.")
        def get_conv_template(name):
            class MockTemplate:
                def __init__(self):
                    self.system_message = system_message
                    self.roles = ['user', 'assistant']
                    self.messages = []
                    self.sep = '###'
                def append_message(self, role, message):
                    if message is not None:
                        self.messages.append((role, message))
                    else:
                        self.messages.append((role, ''))
                def get_prompt(self):
                    # 简化模拟：仅保留系统消息和当前问答对
                    prompt = self.system_message + self.sep + '\n'
                    # 仅处理最后一个用户问题和空白的助手回复
                    user_q = self.messages[-2][1] if len(self.messages) >= 2 else self.messages[0][1]
                    prompt += f'{self.roles[0]}: {user_q}{self.sep}\n'
                    prompt += f'{self.roles[1]}: '
                    return prompt
            return MockTemplate()
    

    # -------------------------------------------------------------------
    # 步骤 1: 预处理和参数检查
    # -------------------------------------------------------------------

    # 模仿 chat 函数中处理单图像问句的逻辑：如果当前有图像，自动添加 <image> 占位符
    if pixel_values is not None and '<image>' not in question:
        question = '<image>\n' + question

    if num_patches_list is None:
        num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
    

    # -------------------------------------------------------------------
    # 步骤 2: 构建对话模板和查询 (query) 字符串
    # -------------------------------------------------------------------
    
    template = get_conv_template(template_name)
    template.system_message = system_message

    # 仅添加当前问题，无历史记录
    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], None) # 模型的回答部分留空
    
    # 获取完整的查询字符串
    query = template.get_prompt()

    # -------------------------------------------------------------------
    # 步骤 3: 图像 token 替换
    # -------------------------------------------------------------------
    
    # 替换查询字符串中的所有 '<image>' 占位符
    for num_patches in num_patches_list:
        # 构造替换的图像 token 字符串
        image_tokens = (
            IMG_START_TOKEN 
            + IMG_CONTEXT_TOKEN * num_image_token * num_patches 
            + IMG_END_TOKEN
        )
        # 只替换第一个 '<image>' 占位符
        query = query.replace('<image>', image_tokens, 1)

    # -------------------------------------------------------------------
    # 步骤 4: Tokenization
    # -------------------------------------------------------------------
    
    # 使用 tokenizer 将最终的 query 字符串转换为 input_ids 和 attention_mask
    model_inputs = tokenizer(query, return_tensors='pt')
    
    input_ids = model_inputs['input_ids']
    attention_mask = model_inputs['attention_mask']

    return input_ids, attention_mask