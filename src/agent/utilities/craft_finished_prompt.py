from typing import Any, Dict, List
from pathlib import Path
import pandas as pd
import json
import os

# ----------------------------- Load Prompt Chain --------------------------------------
def load_json(json_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Loads the result JSON produced by your previous step.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Could not find JSON at: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Expected a dict at the top level of the JSON.")
    return data

def save_to_json(data: Dict[str, Any], filename: str = "src2/parse_subtasks/prompt_chain.json") -> None:
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# ----------------------------- Build Messages -----------------------------------------
def prompt_chain_to_messages(
    prompt_chain: Dict[str, Dict[str, Any]]
) -> Dict[str, List[Dict[str, str]]]:
    """
    Convert a flattened prompt_chain dict to list[dict]
    """
    result: Dict[str, List[Dict[str, str]]] = {}

    for task_name, payload in prompt_chain.items():
        system_text = str(payload.get("system")).strip() or "You are a helpful medical assistant."
        user_text = str(payload.get("user", "")).strip()
        content_text = str(payload.get("content", "")).strip()

        # Fallbacks if tags are missing (should exist from your defaults, but safe-guard anyway)
        start_tag = payload.get("start_tag") or f"[[ ## {task_name} ## ]]"
        end_tag = payload.get("end_tag") or "[[ ## completed ## ]]"

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

        # Include both roles
        result[task_name] = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_message},
        ]

    return result

def load_and_prep_prediction_to_prompt(data:pd.DataFrame,
                                       raw_prompts_data_dict:dict,
                                       sub_prompt_reflection:str,
                                       idx_row:int,
                                       num_best_prompt:int):
  
	# search best prompt
    def _mean(col: str, digits: int = 4) -> str:
        return f"{data[col].mean():.{digits}f}" if col in data.columns else "NA"

    def _mean_len(col: str) -> str:
        return f"{data[col].astype(str).str.len().mean():.0f}" if col in data.columns else "NA"

    
    mt = "LONG_CODES"
    nbp = num_best_prompt
    

    precision = _mean(f"precision_{nbp}_{mt}")
    recall = _mean(f"recall_{nbp}_{mt}")
    f1 = _mean(f"f1_{nbp}_{mt}")

    prompt_len = f"{(data[f'prompt_{nbp}'].astype(str).str.len().mean() - data['TEXT'].astype(str).str.len().mean()):.0f}"

    output_len = _mean_len(f"prediction_{nbp}")
    codes_pred = f"{data[f'predicted_code_{nbp}'].apply(len).mean():.0f}" if f"predicted_code_{nbp}" in data.columns else "NA"

    str_evaluation = (
        f"Precision: {precision}, "
        f"Recall: {recall}, "
        f"F1: {f1}, "
        f"Prompt length: {prompt_len}, "
        f"Output length: {output_len}, "
        f"Mean number codes predicted: {codes_pred}.\n\n"
        )
    
    # get strings from predictions
    str_pred_prompt 	= data.loc[idx_row,f"prompt_{nbp}"]
    str_pred_output 	= data.loc[idx_row,f"prediction_{nbp}"]
    str_pred_codes 		= str(data.loc[idx_row,f"predicted_code_{nbp}"])
    str_ground_truth 	= str(data.loc[idx_row,"LONG_CODES"])
    
    # Cut out system prompt string
    cut_out_sys_str = raw_prompts_data_dict[f"prompt_{nbp}"][0]["content"]
    str_pred_prompt_cut = str_pred_prompt.replace(cut_out_sys_str, "")
    
    
    prompt_reflection = f"""
    {str_pred_prompt_cut}{str_pred_output}<|im_end|>
    Predicted Codes: {str_pred_codes}
    Ground Truth: {str_ground_truth}
    {str_evaluation}
    {sub_prompt_reflection}
    """.strip()
    
    return prompt_reflection



# ----------------------------- Example Main -------------------------------------------
if __name__ == "__main__":
    name_input_json = "10_tasks"
    name_output_json = "10_tasks_prompts"
    path_input = "data/input/"
    path_output = "data/input/"
    prompt_chain = load_json(f"{path_input}{name_input_json}.json")
    result = prompt_chain_to_messages(prompt_chain)

    save_to_json(result, filename = f"{path_output}{name_output_json}.json")
