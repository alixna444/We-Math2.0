# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
import pdb
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import random
from datasets import load_dataset
from jinja2 import Template
from PIL import Image
from PIL.Image import Image as ImageObject
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..models.transformers.qwen2_vl import get_rope_index
from . import torch_functional as VF


def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


class ImageProcessMixin:
    max_pixels: int
    min_pixels: int

    def process_image(self, image: Union[Dict[str, Any], ImageObject]) -> ImageObject:
        if isinstance(image, dict):
            image = Image.open(BytesIO(image["bytes"]))
        elif isinstance(image, bytes):
            image = Image.open(BytesIO(image))

        if (image.width * image.height) > self.max_pixels:
            resize_factor = math.sqrt(self.max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if (image.width * image.height) < self.min_pixels:
            resize_factor = math.sqrt(self.min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image


class RLHFDataset(Dataset, ImageProcessMixin):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        max_pixels: Optional[int] = None,
        min_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = True,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.filter_overlong_prompts = filter_overlong_prompts

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        if os.path.isdir(data_path):
            # when we use dataset builder, we should always refer to the train split
            self.dataset = load_dataset("parquet", data_dir=data_path, split="train")
        elif os.path.isfile(data_path):
            self.dataset = load_dataset("parquet", data_files=data_path, split="train")
        else:
            # load remote dataset from huggingface hub
            self.dataset = load_dataset(data_path, split=data_split)

        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        if self.filter_overlong_prompts:
            self.dataset = self.dataset.filter(self._filter_overlong_prompts, desc="Filtering overlong prompts")
            
        # Initialize index mappings
        self.idx_to_question_id = defaultdict(list)  # There is a one-to-one mapping from idx to question
        self.question_difficulty_to_idx = defaultdict(list)
        self._build_index_mappings()

    def get_question_id(self, idx: int) -> int:
        # get question_id from mathpro
        example = self.dataset[idx]
        if 'question_id' in example:
            return example['question_id']

    def get_difficulty(self, idx: int) -> str:
        # get difficulty from mathpro
        example = self.dataset[idx]
        if 'difficulty' in example:
            return example['difficulty']
        
    def _build_index_mappings(self):
        self.question_difficulty_to_idx = {}
        for idx in range(len(self)):
            question_id = self.get_question_id(idx)
            difficulty = self.get_difficulty(idx)
            self.question_difficulty_to_idx[(question_id, difficulty)] = idx

    def _build_messages(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt_str: str = example[self.prompt_key]
        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            prompt_str = format_prompt.render(content=prompt_str)

        if self.image_key in example:
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            content_list = []
            for i, content in enumerate(prompt_str.split("<image>")):
                if i != 0:
                    content_list.append({"type": "image"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        else:
            return [{"role": "user", "content": prompt_str}]

    def _filter_overlong_prompts(self, example: Dict[str, Any]) -> bool:
        messages = self._build_messages(example)
        processing_class = self.processor if self.processor is not None else self.tokenizer
        return (
            len(processing_class.apply_chat_template(messages, add_generation_prompt=True)) <= self.max_prompt_length
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example: dict = self.dataset[index]
        messages = self._build_messages(example)

        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = [self.process_image(image) for image in example.pop(self.image_key)]
            model_inputs = self.processor(images, [prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"image": images}
            example["multi_modal_inputs"] = dict(model_inputs)
        else:
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]

        if self.processor is not None and self.processor.image_processor.__class__.__name__ == "Qwen2VLImageProcessor":
            # qwen2vl mrope
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw"),
                attention_mask=attention_mask,
            )  # (3, seq_length)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["position_ids"] = position_ids
        example["raw_prompt_ids"] = raw_prompt_ids
        example["ground_truth"] = example.pop(self.answer_key)
        return example


class DynamicDataLoader:
    def __init__(self, dataset, batch_size, shuffle=False, collate_fn=None, num_workers=8):
        """
        Initializes DynamicDataLoader for dynamic batching and scheduling.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn or (lambda x: x)
        self.num_workers = num_workers
        # The scheduler handles which questions to serve next
        self.scheduler = DynamicScheduler(batch_size, dataset)

    def state_dict(self):
        return {}
    
    def __iter__(self):
        return self
        
    def __next__(self):
        # Get indices for the next batch from scheduler
        batch_indices = self.scheduler.get_next_batch(self.dataset)
        if not batch_indices:
            raise StopIteration
            
        batch_samples = []
        for idx in batch_indices:
            sample = self.dataset.__getitem__(idx)
            batch_samples.append(sample)
            
        if self.collate_fn:
            batch_samples = self.collate_fn(batch_samples)
            
        return batch_samples, batch_indices
        
    def update_batch_results(self, batch_indices: List[int], scores: List[List[float]], difficulties: List[str]):
        """
        Update the scheduler with results for each batch sample.
        """
        for idx, score_list, difficulty in zip(batch_indices, scores, difficulties):
            question_id = self.dataset.get_question_id(idx)
            self.scheduler.update_state(question_id, score_list, difficulty)
            
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class QuestionState:
    def __init__(self):
        self.completed = False
        self.current_difficulty = 'base'  # begin from base in every group
        self.wrong_count = 0
        self.last_attempt = None
        self.in_wrong_pool = False
        self.next_difficulty = None

class DynamicScheduler:
    """
    Scheduler for dynamic batch selection based on question state and difficulty.

    Difficulty level meanings:
        - 'a', 'b', 'c', 'd', 'e': corresponding to incremental knowledge point datasets
        - 'x': contextual complexity    
        - 'y': visual complexity              
        - 'z': step complexity                
        - 'xy': both contextual and visual complexity
        - 'xz': contextual and step complexity
        - 'yz': visual and step complexity
        - 'xyz': all three complexity types combined

    The scheduler moves questions across these levels based on response correctness and adaptive policy.
    """
    def __init__(self, batch_size: int, dataset):
        self.batch_size = batch_size
        self.question_states = {}
        self.wrong_pool = []
        self.current_batch = []
        self.difficulty_levels = ['base', 'a', 'b', 'c', 'd', 'e', 'x', 'y', 'z', 'xy', 'xz', 'yz', 'xyz']
        all_question_ids = set(qid for qid, diff in dataset.question_difficulty_to_idx.keys())
        for question_id in all_question_ids:
            self.question_states[question_id] = QuestionState()
        
    def get_next_difficulty(self, current_difficulty: str, is_correct: bool, wrong_count: int) -> str:
        """
        Decide next difficulty to attempt for a question based on current state and answer result.

        Args:
            current_difficulty: The current difficulty attempted.
            is_correct: If the question was answered correctly.
        Returns:
            Next difficulty label, or None if question should be dropped/completed.
        """
        if current_difficulty == 'base':
            next_diff = 'x' if is_correct else None
        elif current_difficulty == 'a':
            next_diff = 'b'
        elif current_difficulty == 'b':
            next_diff = 'c'
        elif current_difficulty == 'c':
            next_diff = 'd'
        elif current_difficulty == 'd':
            next_diff = 'e'
        elif current_difficulty == 'e':
            next_diff = 'x'
        elif current_difficulty == 'x':
            next_diff = 'xy' if is_correct else 'y'
        elif current_difficulty == 'y':
            next_diff = 'xy' if is_correct else None
        elif current_difficulty == 'xy':
            next_diff = 'xz' if is_correct else 'z'

        elif current_difficulty == 'z':
            next_diff = 'xz' if is_correct else None

        elif current_difficulty == 'xz':
            next_diff = 'xyz' if is_correct else 'yz'
        elif current_difficulty == 'yz':
            next_diff = 'xyz' if is_correct else None
        else:
            next_diff = None

        print(f"Current difficulty: {current_difficulty}, Answer {'correct' if is_correct else 'wrong'}, Next: {next_diff}")        
        return next_diff


    def update_state(self, question_id: int, scores: List[float], current_difficulty: str):
        """
        Update state for a given question after model attempted it. Handles transitions and wrong pool.
        """
        if question_id not in self.question_states:
            self.question_states[question_id] = QuestionState()
            
        state = self.question_states[question_id]
        
        # If any score == 1.0 (within rollout.n), consider answered correctly
        is_correct = any(score == 1.0 for score in scores)
        
        if is_correct:
            state.wrong_count = 0
            state.next_difficulty = self.get_next_difficulty(current_difficulty, True, state.wrong_count)
        else:
            state.wrong_count += 1
            state.next_difficulty = self.get_next_difficulty(current_difficulty, False, state.wrong_count)
            if state.next_difficulty == None:
                state.completed = True
                print("Exiting loop - question group completed.")
            else: 
                state.in_wrong_pool = True
                print(f"Question {question_id} added to wrong pool.")
                self.wrong_pool.append((question_id, current_difficulty))
    
        if state.next_difficulty is not None:
            state.current_difficulty = state.next_difficulty
        return state.next_difficulty


    def get_next_batch(self, dataset) -> List[int]:
        """
        Assemble a batch of sample indices dynamically, prioritizing wrong pool questions then normal scheduling.
        """
        batch_indices = []
        while len(batch_indices) < self.batch_size and self.wrong_pool:
            question_id, difficulty = self.wrong_pool.pop(0)
            key = (question_id, difficulty)
            if key in dataset.question_difficulty_to_idx:
                idx = dataset.question_difficulty_to_idx[key]
                print(f"Serving from wrong pool: Qid {question_id}, idx {idx}")
                batch_indices.append(idx)
                
        # 2. Serve normal scheduled questions
        for question_id, state in self.question_states.items():
            if len(batch_indices) >= self.batch_size:
                break
            # Only select questions that are not completed or in wrong pool
            if not state.completed and not state.in_wrong_pool:
                key = (question_id, state.current_difficulty)
                if key in dataset.question_difficulty_to_idx:
                    idx = dataset.question_difficulty_to_idx[key]
                    print(f"Serving normal flow: Qid {question_id}, idx {idx}, current diff {state.current_difficulty}")
                    batch_indices.append(idx)
                            
        # 3. Pad batch if not enough by random sampling
        if len(batch_indices) < self.batch_size:
            remaining = self.batch_size - len(batch_indices)
            print(f"{remaining} remaining, sampling randomly from dataset.")
            random_indices = random.sample(range(len(dataset)), remaining)
            batch_indices.extend(random_indices)
            
        # 4. Truncate to batch_size if exceeded
        if len(batch_indices) > self.batch_size:
            batch_indices = batch_indices[:self.batch_size]
        
        return batch_indices
