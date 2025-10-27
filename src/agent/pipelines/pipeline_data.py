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



def run_pipeline_data(predictor:predict.VLLMPredictor,
                      p_data_config:configs.PipelineDataConfig,
                      p_task_config:configs.PipelineTaskConfig,
                      log: logger_setup.logging.Logger,
                      path_file_output_logs:Path,
                      path_folder_output:Path,
					  path_file_io_data:Path,
					  path_file_io_data_prompts:Path,
                      iteration:int):

    # ---------------------------------------------------------------------------
    # Header: get all required args from config
    # ---------------------------------------------------------------------------

    # Data input
	path_file_io_data		= path_file_io_data
	path_file_io_data_prompts= path_file_io_data_prompts
	iteration               = iteration

    # Model
	name_model              = p_data_config.name_model

    # Inference
	temperature             = p_data_config.temperature 
	max_tokens              = p_data_config.max_tokens
	role_to_add_note        = p_data_config.role_to_add_note
	enable_thinking         = p_data_config.enable_thinking 
	continue_final_message  = p_data_config.continue_final_message 
	add_generation_prompt   = p_data_config.add_generation_prompt 

    # Hardware 
	type_gpu                = p_data_config.type_gpu

    # Data output
	path_folder_output      = path_folder_output

    # ---------------------------------------------------------------------------
    # Process Data: Manipulate some paths and names
    # ---------------------------------------------------------------------------

    # Current Ideration prompt key
	data_prompt_key_iteration = f"prompt_{iteration}"
    

    # ---------------------------------------------------------------------------
    # Run Script: Run pipeline data step by step
    # ---------------------------------------------------------------------------

    # time it
	time_start = time.time()
	log.info("Start pipeline data iteration: %s", iteration)
    
    # Make folder for experiemts today
	path_folder_output.parent.mkdir(parents=True, exist_ok=True)
	path_file_output_logs.parent.mkdir(parents=True, exist_ok=True)
    
    # Load Data
	data = pd.read_parquet(path=path_file_io_data)

    # Load raw prompts
	raw_prompts_json = json_io.load_json(path=path_file_io_data_prompts)
	num_prompts = len(raw_prompts_json)

    # Initialize the tokenizer
	tokenizer = AutoTokenizer.from_pretrained(name_model)

    # Make prompts
	data = add_prompts.make_prompts(df_data=data, 
                                    tokenizer=tokenizer, 
                                    data_prompt_key_iteration=data_prompt_key_iteration,
                                    raw_prompts_json=raw_prompts_json,
                                    role_to_add_note=role_to_add_note,
                                    enable_thinking=enable_thinking, 
                                    continue_final_message=continue_final_message, 
                                    add_generation_prompt=add_generation_prompt)
	time_prompts = time.time()

    # Perform predictions
	data = predictor.generate_model_output_data(df_data=data,
										  		data_prompt_key_iteration=data_prompt_key_iteration,
              									iteration=iteration)

	data.to_parquet(path_file_io_data)
	time_predictions = time.time()

    # Evaluate
	data = evaluation.parse_icd_10_in_outputs(df_data=data, iteration=iteration)
	data = evaluation.make_eval_with_metrics( df_data=data, iteration=iteration, metrics_on="LONG_CODES")
	data = evaluation.make_eval_with_metrics( df_data=data, iteration=iteration, metrics_on="SHORT_CODES")
	data.to_parquet(path_file_io_data)
	time_evaluation = time.time()
    
    # Log time pipeline task
	time_duration = time.time() - time_start
	str_time_p_task_hms = logger_setup._secs_to_hms(time_duration)
	f1_col_current_prompt = f"f1_{iteration}_LONG_CODES"
	f1_long_prompt_0 = f"{data[f1_col_current_prompt].mean():.{4}f}"
	log.info("Finished pipeline data iteration: %s, duration: %s, F1_0_Long: %s", iteration, str_time_p_task_hms, f1_long_prompt_0)

    # Print experiment meta data
	logger_setup.pipeline_data_log_and_report(
    p_data_config=p_data_config,
    p_task_config=p_task_config,
    data=data,
    path_folder_output=path_folder_output,
    num_prompts=num_prompts,
    time_start=time_start, 
    time_prompts=time_prompts, 
    time_predictions=time_predictions, 
    time_evaluation=time_evaluation,
    iteration=iteration,
    log=log,
    log_level="INFO",
    metrics_types=("LONG_CODES", "SHORT_CODES"),
    text_column="TEXT",
    )

if __name__ == "__main__":
    run_pipeline_data()