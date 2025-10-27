import re
import pandas as pd


# -------------------------------------- Functions ---------------------------------------------
def _parse_icd_10_code_from_str(text: str, pattern: str = r'\b[A-Z][0-9]{2}(?:\.[A-Z0-9]{1,4}|[A-Z0-9]{1,4})?\b'): # r'\b[A-Z]\d{2}\.\d+[A-Z]*\b'
    """Search for icd 10 code in string and return unique codes as list"""
    matches = re.findall(pattern, text)
    unique_matches = list(dict.fromkeys(matches))
    unique_matches = [u_match.replace(".", "") for u_match in unique_matches]
    return unique_matches


def parse_icd_10_in_outputs(df_data: pd.DataFrame, iteration:int) -> pd.DataFrame:
    """parse columns and put result in new column"""
    df_data[f"predicted_code_{iteration}"] = df_data[f"prediction_{iteration}"].apply(_parse_icd_10_code_from_str)
    return df_data


def _apply_metrics(row: pd.DataFrame, pred_col: str, metrics_on: str) -> pd.Series:
    """
    Calculate precision, recall, F1
    """
    if metrics_on == "LONG_CODES":
        predictions = set(row[pred_col])
        ground_truth    = set(row[metrics_on])
        
    elif metrics_on == "SHORT_CODES":
        predictions = set([icd_code[:4] for icd_code in row[pred_col]])
        ground_truth    = set(row[metrics_on])
    else:
        print("Error with LONG or SHORT Codes. No metric applied!")
    
    # True predictions
    true_positives  = len(ground_truth.intersection(predictions))
    
    # add columns with precision, recall, f1
    precision       = true_positives / len(predictions) if len(predictions) else 0
    recall          = true_positives / len(ground_truth) if len(ground_truth) else 0
    f1              = 2 * (precision * recall) / (precision + recall) if precision and recall else 0

    return pd.Series({'precision': precision, "recall": recall, "f1": f1})


def make_eval_with_metrics(df_data:pd.DataFrame, iteration:int, metrics_on:str= "LONG_CODES") -> pd.DataFrame:
    """
    Apply metric to predicted codes and make column precision, recall, F1.
    """
    if isinstance(df_data.iloc[0, 2], str):
        df_data[metrics_on] = df_data[metrics_on].apply(lambda x: x.strip("[]").replace("'", "").split())
    # for num_prompt in range(num_prompts):
    df_data[[f'precision_{iteration}_{metrics_on}', f'recall_{iteration}_{metrics_on}', 
                f'f1_{iteration}_{metrics_on}']] = df_data.apply(_apply_metrics,
                                                        pred_col=f"predicted_code_{iteration}",
                                                        metrics_on=metrics_on,
                                                        axis=1)
    return df_data


def pick_best_prompt(data: pd.DataFrame, iteration: int):
    """Calculate the best prompt with LONG_CODES metric and return the prompt index."""
    
    def _mean(col: str, digits: int = 4) -> str:
        return f"{data[col].mean():.{digits}f}" if col in data.columns else "NA"
    mt = "LONG_CODES"
    score_dict = {}
    
    if iteration > 0:
        for num_prompt_i in range(iteration):
            f1 = _mean(f"f1_{num_prompt_i}_{mt}")
            score_dict[f1] = num_prompt_i
        
        best_score = max(score_dict.keys())
        num_best_prompt = score_dict[best_score]
    else:
        num_best_prompt = 0
        
    return num_best_prompt