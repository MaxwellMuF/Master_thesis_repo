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



def run_pipeline_task_2(predictor:predict.VLLMPredictor,
    				  p_data_config:configs.PipelineDataConfig,
                      p_task_config:configs.PipelineTaskConfig,
                      log: logger_setup.logging.Logger,
                      path_file_output_logs:Path,
                      path_folder_output:Path,
                      path_file_io_data:Path,
                      path_file_io_data_prompts:Path,
                      path_file_io_prompts_task_1:Path,
                      path_file_io_prompts_task_2:Path,
                      num_best_prompt:int,
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
    log.info("Start pipeline task_2 iteration: %s", iteration)

    # Load task data containing task prompt
    dict_task_1 = json_io.load_json(path=path_file_io_prompts_task_1)
    dict_task_2 = json_io.load_json(path=path_file_io_prompts_task_2)
    
    # Current Ideration task key
    task_key_temp = list(dict_task_2.keys())[0]
    task_key_temp = task_key_temp.replace("_0", "")
    task_key_iteration = task_key_temp + f"_{iteration}"
    task_key_next_iteration = task_key_temp + f"_{iteration+1}"
    task_key_best_prompt = task_key_temp + f"_{num_best_prompt}"
    data_prompt_key_best_prompt = f"prompt_{num_best_prompt}"

    # Load raw prompts: data prompt
    raw_prompts_data_dict = json_io.load_json(path=path_file_io_data_prompts)

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(name_model)

    ## Make prompts ##
    # copy task for next iteration
    task_template = dict_task_2[task_key_iteration].copy()
    dict_task_2 = {**dict_task_2, 
                   task_key_next_iteration : task_template}
        
    # add last user prompt and last system prompt
    dict_task_2[task_key_iteration]["user"] = dict_task_2[task_key_best_prompt]["user"].format(
        raw_prompts_data_dict[data_prompt_key_best_prompt][1]["content"],
        raw_prompts_data_dict[data_prompt_key_best_prompt][0]["content"]
    )
    
    # Add refelction prompt
    dict_task_2 = add_prompts.data_task_to_messages(data_tasks=dict_task_2,
                                                    task_key_iteration=task_key_iteration,
                                                    tokenizer=tokenizer,
                                                    enable_thinking=enable_thinking,
                                                    continue_final_message=continue_final_message,
                                                    add_generation_prompt=add_generation_prompt)

    # Perform predictions
    dict_task_2 = predictor.generate_model_output_tasks(data_tasks=dict_task_2,
                                                        task_key_iteration=task_key_iteration)

    # Parse result and save it
    dict_task_2 = parse_text_between_tags.parse_task_data(data_tasks=dict_task_2, 
                                                          task_key_iteration=task_key_iteration)
    json_io.save_json(json_data=dict_task_2, path=path_file_io_prompts_task_2)
    
    # Make and save new data prompt json
    idx_new_prompt = iteration + 1
    raw_prompts_data_dict = add_prompts.make_new_data_prompt_json_system(
                                        raw_prompts_data_dict=raw_prompts_data_dict,
        								prompt_str=dict_task_2[task_key_iteration]["parsed_output"],
                						idx_new_prompt=idx_new_prompt)
    json_io.save_json(json_data=raw_prompts_data_dict, path=path_file_io_data_prompts)

    # Log time pipeline task
    time_duration = time.time() - time_start
    str_time_p_task_hms = logger_setup._secs_to_hms(time_duration)
    log.info("Finished pipeline task_2 iteration: %s, duration: %s", iteration, str_time_p_task_hms)

if __name__ == "__main__":
    run_pipeline_task_2()