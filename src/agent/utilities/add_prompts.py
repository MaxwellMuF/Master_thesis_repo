import pandas as pd
from transformers import AutoTokenizer
from typing import Any, Dict, List
from pathlib import Path
import os


# -------------------------------------- Functions ---------------------------------------------
def _make_chat_template(tokenizer, message: list[dict], enable_thinking: bool,
                       continue_final_message: bool=False, add_generation_prompt: bool=True):
    """Chat template for model prediction. Return one string all roles and contents"""
    text = tokenizer.apply_chat_template(
        message,
        tokenize=False,
        continue_final_message=continue_final_message,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking
        )
    return text


def make_prompts(df_data:pd.DataFrame, tokenizer, data_prompt_key_iteration:str, raw_prompts_json:dict,
                 role_to_add_note:str="user",
                 enable_thinking:bool=True, continue_final_message:bool=False, add_generation_prompt:bool=True
                  ) -> pd.DataFrame:
    """Go throu all prompt texts and add admission notes"""
    # for prompt_key, prompt_value in raw_prompt.items():
    prompt_key   = data_prompt_key_iteration
    prompt_value = raw_prompts_json[prompt_key]
    
    df_data[prompt_key] = df_data["TEXT"].apply(
        lambda x: _make_chat_template(
            tokenizer,
            message=[
                (
                    # add note if condition
                    {**roles, "content": roles["content"].format(x)}
                    if roles["role"] == role_to_add_note
                    else roles
                )
                for roles in prompt_value 
            ],
            enable_thinking=enable_thinking,
            continue_final_message=continue_final_message,
            add_generation_prompt=add_generation_prompt
        )
    )
    return df_data


def make_prompts_tasks_old(data_tasks:dict[str,dict[str,str]], tokenizer, 
                       enable_thinking:bool=True, continue_final_message:bool=False, 
                       add_generation_prompt:bool=True
                  ) -> pd.DataFrame:
    """"""
    for prompt_key, prompt_value in data_tasks.items():
        message_keys = ["system","user"] 
        message_current = []
        for p_role in message_keys:
            message_current.append({
                "role" : p_role,
                "content" : prompt_value[p_role]
            })
        data_tasks[prompt_key]["raw_input"] = _make_chat_template(
                tokenizer,
                message=message_current,
                enable_thinking=enable_thinking,
                continue_final_message=continue_final_message,
                add_generation_prompt=add_generation_prompt
            )
    return data_tasks


def data_task_to_messages(data_tasks: Dict[str, Dict[str, Any]],
                          task_key_iteration:str,
                           tokenizer: Any,
                           enable_thinking: bool,
                           continue_final_message: bool,
                           add_generation_prompt: bool) -> Dict[str, List[Dict[str, str]]]:
    """
    adjust the data_task file with chat template
    """
    prompt_dict = {}
    task_key = task_key_iteration
    task_value = data_tasks[task_key]
    
    # for task_key, task_value in data_tasks.items():
    system_text = str(task_value.get("system")).strip() or "You are a helpful medical assistant."
    user_text = str(task_value.get("user", "")).strip()
    content_text = str(task_value.get("content", "")).strip()

    # Fallbacks if tags are missing (should exist from your defaults, but safe-guard anyway)
    start_tag = task_value.get("start_tag") or f"[[ ## {task_key} ## ]]"
    end_tag = task_value.get("end_tag") or "[[ ## completed ## ]]"

    # Build the user message
    parts: List[str] = []
    if user_text:
        parts.append(user_text)
    if content_text:
        parts.append(f"Additional context:\n{content_text}")

    # Final instruction using the tags
    parts.append(
        f"Think through this step by step and then give a final answer. "
        f"Write your final answer between two tags starting with {start_tag} then your final answer and ending with {end_tag}."
    )

    user_message = "\n\n".join([p for p in parts if p.strip()])

    # Always include both roles to satisfy downstream template expectations
    prompt_dict = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_message},
    ]
    
    data_tasks[task_key]["raw_input"] = _make_chat_template(
            tokenizer,
            message=prompt_dict,
            enable_thinking=enable_thinking,
            continue_final_message=continue_final_message,
            add_generation_prompt=add_generation_prompt
        )

    return data_tasks

def make_new_data_prompt_json(prompt_str: str, idx_new_prompt: int) -> Dict:
    """Make dict with roles for chat temp transformers"""
    prompt_dict = {
    f"prompt_{idx_new_prompt}": [
        {
        "role": "system",
        "content": "You are a helpful medical assistant."
        },
        {
        "role": "user",
        "content": f"{prompt_str} \n\nHere is the patient's admission note"
        ":\n\n{}"
        }
        ]
    }
    return prompt_dict

def make_new_data_prompt_json_system(raw_prompts_data_dict:dict, prompt_str: str, idx_new_prompt: int) -> Dict:
    """Make dict with roles for chat temp transformers"""
    
    raw_prompts_data_dict[f"prompt_{idx_new_prompt}"][0]["content"] = prompt_str
    
    return raw_prompts_data_dict
