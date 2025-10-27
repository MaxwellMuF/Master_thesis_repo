import os
import time
import pandas as pd
from datetime import datetime
from pathlib import Path
from transformers import AutoTokenizer

from agent.helper       import configs
from agent.helper       import logger_setup
from agent.helper       import json_io

from agent.utilities    import predict
from agent.utilities    import add_prompts
from agent.utilities    import evaluation
from agent.utilities    import parse_text_between_tags
from agent.utilities    import craft_finished_prompt



def run_pipeline_task(predictor:predict.VLLMPredictor,
    				  p_data_config:configs.PipelineDataConfig,
                      p_task_config:configs.PipelineTaskConfig,
                      log: logger_setup.logging.Logger,
                      path_file_output_logs:Path,
                      path_folder_output:Path,
                      path_file_io_data:Path,
                      path_file_io_data_prompts:Path,
                      path_file_io_prompts_task_1:Path,
                      path_file_io_prompts_task_2:Path,
                      iteration:int):

    # ---------------------------------------------------------------------------
    # Header: get all required args from config
    # ---------------------------------------------------------------------------

    # Data input: last iteration pipeline data output
    path_file_io_data           = path_file_io_data
    path_file_io_data_prompts   = path_file_io_data_prompts
    path_file_io_prompts_task_1 = path_file_io_prompts_task_1
    path_file_io_prompts_task_2 = path_file_io_prompts_task_2
    
    random_seed				= p_task_config.random_seed

    # Model
    name_model              = p_data_config.name_model

    # Inference
    temperature             = p_data_config.temperature 
    max_tokens              = p_data_config.max_tokens

    enable_thinking         = p_data_config.enable_thinking
    continue_final_message  = p_data_config.continue_final_message 
    add_generation_prompt   = p_data_config.add_generation_prompt

    # Hardware 


    # Data output
    path_folder_output      = path_folder_output

    # ---------------------------------------------------------------------------
    # Run Script: Run pipeline data step by step
    # ---------------------------------------------------------------------------

    # time it
    time_start = time.time()
    log.info("Start pipeline task iteration: %s", iteration)

    # Load task data containing task prompt
    dict_task_1 = json_io.load_json(path=path_file_io_prompts_task_1)
    
    # Current Ideration task key
    task_key_temp = list(dict_task_1.keys())[0]
    task_key_temp = task_key_temp.replace("_0", "")
    task_key_iteration = task_key_temp + f"_{iteration}"
    task_key_next_iteration = task_key_temp + f"_{iteration+1}" 

    # Load raw prompts: data prompt
    raw_prompts_data_dict = json_io.load_json(path=path_file_io_data_prompts)

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(name_model)

    # Make prompts
    task_template = dict_task_1[task_key_iteration].copy()
    dict_task_1 = {**dict_task_1, 
                   task_key_next_iteration : task_template}
    dict_task_1 = add_prompts.data_task_to_messages(data_tasks=dict_task_1,
                                                    task_key_iteration=task_key_iteration,
                                                    tokenizer=tokenizer,
                                                    enable_thinking=enable_thinking,
                                                    continue_final_message=continue_final_message,
                                                    add_generation_prompt=add_generation_prompt)

	# load predictions
    data = pd.read_parquet(path=path_file_io_data)

    # Add refelction prompt
    random_seed_current = random_seed + iteration
    idx_row = data.sample(n=1, random_state=random_seed_current).index[0]
    log.info("Use seed: %s, data_idx: %s, hadm_id: %s", random_seed_current, idx_row, data.loc[idx_row,"hadm_id"])
    
    # Pick prompt with highest score
    num_best_prompt = evaluation.pick_best_prompt(data=data, iteration=iteration)

    # Craft prompt with best prompt and random sample note
    dict_task_1[task_key_iteration]["raw_input"] = craft_finished_prompt.load_and_prep_prediction_to_prompt(
        										data=data,
                                                raw_prompts_data_dict=raw_prompts_data_dict,
                                                sub_prompt_reflection=dict_task_1[task_key_iteration]["raw_input"],
                                                idx_row=idx_row,
                                                num_best_prompt=num_best_prompt)

    # Perform predictions
    dict_task_1 = predictor.generate_model_output_tasks(data_tasks=dict_task_1,
                                                        task_key_iteration=task_key_iteration)

    # Parse result and save it
    dict_task_1 = parse_text_between_tags.parse_task_data(data_tasks=dict_task_1,
                                                          task_key_iteration=task_key_iteration)
    json_io.save_json(json_data=dict_task_1, path=path_file_io_prompts_task_1)
    
    # Make and save new data prompt json
    idx_new_prompt = iteration + 1
    new_data_prompt_dict = add_prompts.make_new_data_prompt_json(
        								prompt_str=dict_task_1[task_key_iteration]["parsed_output"],
                						idx_new_prompt=idx_new_prompt)
    raw_prompts_data_dict = {**raw_prompts_data_dict, **new_data_prompt_dict}
    json_io.save_json(json_data=raw_prompts_data_dict, path=path_file_io_data_prompts)

    

    # Log time pipeline task
    time_duration = time.time() - time_start
    str_time_p_task_hms = logger_setup._secs_to_hms(time_duration)
    log.info("Finished pipeline task iteration: %s, duration: %s", iteration, str_time_p_task_hms)
    
    return num_best_prompt

if __name__ == "__main__":
    run_pipeline_task()