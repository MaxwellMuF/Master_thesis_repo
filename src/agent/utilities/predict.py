import json
import pandas as pd
from vllm import LLM, SamplingParams
from typing import Optional


class VLLMPredictor:
    def __init__(
        self,
        name_model: str,
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
        temperature: float = 0.6,
        max_tokens: int = 10_000,
        top_p: float = 0.95,
        top_k: int = 20,
    ):
        # Load once
        self.llm = LLM(model=name_model, dtype=dtype, trust_remote_code=trust_remote_code)

        # Store defaults
        self._default_params = dict(
            temperature=temperature, top_p=top_p, top_k=top_k, max_tokens=max_tokens
        )
        # Build default sampling params
        self.sampling_params = SamplingParams(**self._default_params)

    def _sampling_params_with_overrides(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> SamplingParams:
        """Return self.sampling_params unless any override"""
        if (
            temperature is None
            and max_tokens is None
            and top_p is None
            and top_k is None
        ):
            return self.sampling_params

        merged = self._default_params.copy()
        if temperature is not None: 
            merged["temperature"] = temperature
        if max_tokens is not None:  
            merged["max_tokens"]  = max_tokens
        if top_p is not None:       
            merged["top_p"]       = top_p
        if top_k is not None:       
            merged["top_k"]       = top_k
        return SamplingParams(**merged)

    def generate_model_output_data(
        self,
        df_data: pd.DataFrame,
        data_prompt_key_iteration:str,
        iteration:int,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> pd.DataFrame:
        """prediction_{i} columns from prompt_{i} columns."""
        sp = self._sampling_params_with_overrides(
            temperature=temperature, max_tokens=max_tokens, top_p=top_p, top_k=top_k
        )


        col = data_prompt_key_iteration
        if col not in df_data.columns:
            raise KeyError(f"Missing column: {col}")
        inputs = df_data[col].tolist()
        outputs = self.llm.generate(inputs, sp)
        df_data[f"prediction_{iteration}"] = [o.outputs[0].text for o in outputs]
        return df_data

    def generate_model_output_tasks(
        self,
        data_tasks: dict[str, dict[str, str]],
        task_key_iteration: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        input_key: str = "raw_input",
        output_key: str = "raw_output",
    ) -> dict[str, dict[str, str]]:
        """
        Takes a mapping
        """
        sp = self._sampling_params_with_overrides(
            temperature=temperature, max_tokens=max_tokens, top_p=top_p, top_k=top_k
        )

        # Current Task
        task_key = task_key_iteration 

        # check if keys exist in data_tasks
        if task_key not in data_tasks.keys():
            raise KeyError(f"Missing key in tasks_dict: {task_key}")
        if input_key not in data_tasks[task_key].keys():
            raise KeyError(f"Missing key in task {task_key}: {input_key}")

        input_list = [data_tasks[task_key][input_key]]
        # Generate model preds and save output in data_tasks
        outputs = self.llm.generate(input_list, sp)
        texts = [o.outputs[0].text for o in outputs]
        data_tasks[task_key][output_key] = texts[0]

        return data_tasks

