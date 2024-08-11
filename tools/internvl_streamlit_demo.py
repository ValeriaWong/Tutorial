import json
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
import torchvision.transforms as T
from transformers import AutoTokenizer, AutoModel
import copy
import warnings
from dataclasses import asdict, dataclass
from typing import Callable, List, Optional

import streamlit as st
import torch
from torch import nn
from transformers.generation.utils import (LogitsProcessorList,
                                           StoppingCriteriaList)
from transformers.utils import logging

from transformers import AutoTokenizer, AutoModelForCausalLM  # isort: skip

logger = logging.get_logger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


model_name_or_path = "/root/models/Mini-InternVL-Chat-2B-V1-5"

# model_name_or_path = "/root/model_trained/work_dirs/internvl_ft/run_7_hf"


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size),
                 interpolation=InterpolationMode.BICUBIC),
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


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=6):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values



@torch.inference_mode()
def inference(model,tokenizer, image_path, question, generation_config):
    pixel_values = load_image(
        image_path, input_size=448, max_num=6).to(torch.bfloat16).cuda()
    # tokenizer = AutoTokenizer.from_pretrained(
    #     model_name_or_path, trust_remote_code=True)

    # model = AutoModel.from_pretrained(
    #     model_name_or_path,
    #     torch_dtype=torch.bfloat16,
    #     low_cpu_mem_usage=True,
    #     trust_remote_code=True).eval().cuda()
    response, history = model.chat(
        tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
    # results = {}
    # results[model_path] = response

    yield response


@dataclass
class GenerationConfig:
    # this config is used for chat to provide more diversity
    max_length: int = 4096
    top_p: float = 0.75
    temperature: float = 0.1
    do_sample: bool = False
    repetition_penalty: float = 1.000





def on_btn_click():
    del st.session_state.messages


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True)

    model = AutoModel.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True).eval().cuda()
    return model, tokenizer



user_prompt = '<|im_start|>user\n{user}<|im_end|>\n'
robot_prompt = '<|im_start|>assistant\n{robot}<|im_end|>\n'
cur_query_prompt = '<|im_start|>user\n{user}<|im_end|>\n\
    <|im_start|>assistant\n'


def combine_history(prompt):
    messages = st.session_state.messages
    meta_instruction = ('')
    total_prompt = f"<s><|im_start|>system\n{meta_instruction}<|im_end|>\n"
    for message in messages:
        cur_content = message['content']
        if message['role'] == 'user':
            cur_prompt = user_prompt.format(user=cur_content)
        elif message['role'] == 'robot':
            cur_prompt = robot_prompt.format(robot=cur_content)
        else:
            raise RuntimeError
        total_prompt += cur_prompt
    total_prompt = total_prompt + cur_query_prompt.format(user=prompt)
    return total_prompt


def render_response(cur_response):
    try:
        # 尝试将响应内容解析为 JSON
        json_response = json.loads(cur_response)
        st.json(json_response)  # 如果成功解析为 JSON，则使用 st.json 渲染
    except json.JSONDecodeError:
        # 如果解析失败，检查是否可能是 Markdown 格式
        if any(tag in cur_response for tag in ['#', '*', '-', '```']):
            st.markdown(cur_response)  # 使用 markdown 渲染
        else:
            st.code(cur_response)  # 否则使用 code 渲染


def main():
    # torch.cuda.empty_cache()
    print('load model begin.')
    model, tokenizer = load_model()
    print('load model end.')

    st.title('InternVL-Chat-2B-V1-5')

    # generation_config = prepare_generation_config()
    generation_config = {
        "num_beams": 1,
        "max_new_tokens": 4096,
        "do_sample": False
    }

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message['role'], avatar=message.get('avatar')):
            st.markdown(message['content'])

    # Accept user input
    image_file = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png"])

    # 创建两个列
    col1, col2 = st.columns([1, 3])

    with col1:
        if image_file is not None:
            st.image(image_file, caption="Uploaded Image",
                    use_column_width=False, width=150)
        else:
            st.warning("Please upload an image.")

    with col2:
        question = st.text_input("Enter your question:")
    
    if st.button("Submit"):
        if image_file and question:
            with st.chat_message('user'):
                st.markdown(question)
            real_prompt = combine_history(question)
            st.session_state.messages.append({
                'role': 'user',
                'content': question,
            })
            with st.chat_message('robot'):
                message_placeholder = st.empty()
                for cur_response in inference(
                        model,tokenizer, image_file, question, generation_config
                ):  
                    message_placeholder.markdown(cur_response + '▌')
                message_placeholder.empty()  # 清除临时占位符
                render_response(cur_response)
            st.session_state.messages.append({
                'role': 'robot',
                'content': cur_response,
            })
            torch.cuda.empty_cache()
        else:
            st.warning("Please upload an image and enter a question.")

    # question = "请从这张聊天截图中提取结构化信息"  # 提问
    # prompt = '''请从这张聊天截图中提取结构化信息,格式如下：
    # ```
    # {
    #     "dialog_name": "<dialog_name>",
    #     "conversation": [
    #         {
    #             "timestamp": "<timestamp>",
    #             "speaker": "<speaker>",
    #             "content": "<content>",
    #             "image": "<image>",
    #             "transfer": [],
    #             "file": []
    #         },
    #     ...
    #     ]
    # }
    # ```
    # '''
   


if __name__ == '__main__':
    main()
