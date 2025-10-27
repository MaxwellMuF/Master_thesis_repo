import shutil
from pathlib import Path
from datetime import datetime

from agent.helper       import logger_setup
from agent.helper       import configs

from agent.utilities    import predict
from agent.utilities    import evaluation

from agent.pipelines    import pipeline_data
from agent.pipelines    import pipeline_task
from agent.pipelines    import pipeline_task_2

# -------------------------------------- Run Script ---------------------------------------------

# Parse arguments from yaml and build config data class (dict like)
p_data_config, p_task_config = configs.parse_and_build_configs()

# Make Folders: parent = experiment today, child = iteration
name_model_shot = Path(p_data_config.name_model).name
path_folder_output = Path(p_data_config.path_output) / name_model_shot / f'{datetime.now().date()}-{str(datetime.now().replace(microsecond=0).time()).replace(":","-")}'
path_file_output_logs = path_folder_output / "logs" / f"pipeline_data.log"
path_file_output_logs.parent.mkdir(parents=True, exist_ok=True)

# setup logger
log = logger_setup.setup_logging(level="INFO", 
                                 log_file=path_file_output_logs)

# Init LLM model
predictor = predict.VLLMPredictor(name_model=p_data_config.name_model,
                                  temperature=p_data_config.temperature,
                                  max_tokens=p_data_config.max_tokens) 

## Copy prompts (data and task) to experiment folder
# Make output prompts folder
path_folder_output_prompts = path_folder_output / "prompts"
path_folder_output_prompts.mkdir(parents=True, exist_ok=True)

# Copy prompts (data and tasks) to output prompts folder
path_folder_startprompts = Path(p_data_config.path_prompt_set)
path_file_prompt_data_start = path_folder_startprompts / p_data_config.name_prompts_data
path_file_prompt_task_1_start = path_folder_startprompts / p_task_config.name_prompt_task_1
path_file_prompt_task_2_start = path_folder_startprompts / p_task_config.name_prompt_task_2
shutil.copy2(src=path_file_prompt_data_start,   dst=path_folder_output_prompts / p_data_config.name_prompts_data)
shutil.copy2(src=path_file_prompt_task_1_start, dst=path_folder_output_prompts / p_task_config.name_prompt_task_1)
shutil.copy2(src=path_file_prompt_task_2_start, dst=path_folder_output_prompts / p_task_config.name_prompt_task_2)

# Data and prompts path files input output (io)
path_file_io_data           = path_folder_output / "data" / p_data_config.name_data
path_file_io_data_prompts   = path_folder_output_prompts / p_data_config.name_prompts_data
path_file_io_prompts_task_1 = path_folder_output_prompts / p_task_config.name_prompt_task_1
path_file_io_prompts_task_2 = path_folder_output_prompts / p_task_config.name_prompt_task_2

# Copy data to output folder data
path_file_io_data.parent.mkdir(parents=True, exist_ok=True)
path_file_input_data = Path(p_data_config.path_data) / p_data_config.name_data
shutil.copy2(src=path_file_input_data, dst=path_file_io_data)


# ---------------------------------------------------------------------------
# Start Agent: iterativ iterations
# ---------------------------------------------------------------------------
num_iterations = p_task_config.max_iterations
for iteration in range(num_iterations):
    # Log info iteration
    log.info("Start iteration: %s", iteration)
    
    # Run data pipeline
    pipeline_data.run_pipeline_data(predictor=predictor,
                                    p_data_config=p_data_config,
                                    p_task_config=p_task_config,
                                    log=log,
                                    path_file_output_logs=path_file_output_logs,
                                    path_folder_output=path_folder_output,
                                    path_file_io_data=path_file_io_data,
                                    path_file_io_data_prompts=path_file_io_data_prompts,
                                    iteration=iteration)

    if iteration < num_iterations-1 and p_task_config.improve_prompt:
        # Run task_1 pipeline: new user prompt 
        num_best_prompt = pipeline_task.run_pipeline_task(predictor=predictor,
                                        p_data_config=p_data_config,
                                        p_task_config=p_task_config,
                                        log=log,
                                        path_file_output_logs=path_file_output_logs,
                                        path_folder_output=path_folder_output,
                                        path_file_io_data=path_file_io_data,
                                        path_file_io_data_prompts=path_file_io_data_prompts,
                                        path_file_io_prompts_task_1=path_file_io_prompts_task_1,
                                        path_file_io_prompts_task_2=path_file_io_prompts_task_2,
                                        iteration=iteration)
        
        # Run task_2 pipeline: new user prompt
        pipeline_task_2.run_pipeline_task_2(predictor=predictor,
                                            p_data_config=p_data_config,
                                            p_task_config=p_task_config,
                                            log=log,
                                            path_file_output_logs=path_file_output_logs,
                                            path_folder_output=path_folder_output,
                                            path_file_io_data=path_file_io_data,
                                            path_file_io_data_prompts=path_file_io_data_prompts,
                                            path_file_io_prompts_task_1=path_file_io_prompts_task_1,
                                            path_file_io_prompts_task_2=path_file_io_prompts_task_2,
                                            num_best_prompt=num_best_prompt,
                                            iteration=iteration)
log.info("Agent finished. Output folder: %s", path_folder_output)