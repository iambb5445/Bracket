import json
import os
import argparse
from llm_connector import OpenAIChat, LLMConnector
import pandas as pd
import time
from utility import get_safe_filename
from joblib import delayed, Parallel
from tqdm import tqdm

thread_count: int|None = None

def get_submission_info(filename:str|None):
    info: dict[str, str|int|None] = {'student_id': None, 'late': None}
    if filename is None:
        return info
    basename = os.path.basename(filename)
    parts = basename.split('_')
    info['student_id'] = parts[0]
    info['late'] = 1 if 'LATE' in parts else 0
    return info

def analyze_criteria(chat, submission, criteria, log_id: int|None):
    with open(os.path.join(submissions_dir, submission), 'r', encoding="utf8") as file:
        submission_data = file.read()
    results = {}
    chat.inject(LLMConnector.Role.User, "The student has submitted the following:\n---submission-start---\n" + submission_data + "\n---submission-end---\n")
    # fill submission metadata (student id, etc.)
    submission_metadata = get_submission_info(submission)
    results.update(submission_metadata)
    # fill criteria
    for point in criteria:
        name = point.get('name', 'Unknown')
        method = point.get('method', 'None')
        if method == 'match':
            assert 'value' in point, f'\"match\" criteria with no value to match for {name}'
            # TODO [important] search for this in Brace's repsonse only!
            result = 1 if point['value'] in submission_data else 0
        elif method == 'LLM':
            result = chat.ask(point['prompt'])
        else:
            raise Exception(f'Unknown grading method: {method}')
        results[name] = result
    if log_id is not None:
        chat.dump(f"{output_filename}_{log_id}.log")
    return results

if __name__=="__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('submissions_dir', type=str, help="Directory containing submissions downloaded from Canvas")
    parser.add_argument('rubric_filename', type=str, help="The json file containing the rubric for the assignment")
    parser.add_argument('--test', action='store_true', help="Use this option to run only on the first 3 submission (used for testing purposes)")
    args = parser.parse_args()
    # read submissions filenames
    submissions_dir = args.submissions_dir
    submissions = os.listdir(submissions_dir)
    if args.test:
        submissions = submissions[:3]
    # parse rubric
    rubric_filename = args.rubric_filename
    with open(rubric_filename, 'r', encoding="utf8") as f:
        rubric = json.load(f)
    properties = rubric.get("properties", {})
    llm_properties = properties.get("LLM", {})
    criteria = rubric.get("criteria", [])
    # find the criteria in submissions
    chat = OpenAIChat(OpenAIChat.OpenAIModel.GPT_4O_mini, llm_properties.get("system", None))
    output_filename = get_safe_filename(f"{os.path.basename(rubric_filename)}_{chat.openAI_model}_{int(time.time())}")
    results = []
    if thread_count is None:
        results = [analyze_criteria(chat.copy(), submissions[i], criteria, i) for i in tqdm(range(len(submissions)))]
    else:
        results = Parallel(n_jobs=thread_count)(delayed(analyze_criteria)(chat.copy(), submissions[i], criteria, i) for i in tqdm(range(len(submissions))))
    # create pandas df
    columns = list(get_submission_info(None).keys()) + [c.get('name', 'unknown') for c in criteria]
    grades = dict([(c, []) for c in columns])
    for result in results:
        for k in grades.keys():
            # don't raise error at this point
            # We are already done with our calculations, might as well log what we have.
            if result is None :
                print(f"Error: result is none")
                result = {}
            elif k not in result:
                print(f"Error: cannot find {k} in criteria responses") 
            grades[k].append(result.get(k, 'ERROR'))
    df = pd.DataFrame(grades)    
    # write in file
    print(f"{submissions_dir} graded based on {rubric_filename} using {chat.openAI_model}")
    #   token count does not work with parallel runs
    #   since the state of the program is copied and not shared between threads
    print(f"Total token count: {OpenAIChat.TOTAL_TOKEN_COUNT}")
    df.to_csv(f"grades_{output_filename}.csv")
        