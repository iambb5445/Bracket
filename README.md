# Bracket

Bracket is a simple grader tool aimed to use for grading feedback on Canvas submission and more specificaly Brace.

## How to use

### Requirements

Requirements can be installed with pip using `requirements.txt`. Using a virtualenv is encouraged but not required.

```bash
pip install -r requirements
```

## Authentication Key

To access the LLM features of Bracket, add your OpenAI API key to `auth.py`. Your API key can be found at https://platform.openai.com/api-keys.

## Rubric Format

Running Bracket requires a rubric file specifying how each submission should be processed. The rubric should be a `json` file with the following format:

- "properties": Describes the system-wide properties used for analyzing the assignment
  - "LLM":
    - "system": System prompt used every time the LLM is used.
- "criteria": A **list** of criteria points. The final output file includes one column per each criteria. Each item in the list describes how the values in that column are calculated.
  - "name": The name of the criteria. Used as the column name in the final csv file and for logging purposes (default: "unknown")
  - "method": The method through which this criteria is calculated. Currently only support "match" (finding whether or not a keyword exists in the submission) or "LLM" (using LLM to calculate the criteria).
  - "value": `"match" only` The value that the "match" criteria searchs for.
  - "prompt": `"LLM" only` The prompt that is sent to the LLM along with the submission. All llm queries are in the same conversation and in the same order defined in criteria.

An example of the rubric can be found at `L03-rubric.jason`.

## Run

To run the code, use the following command:

```
usage: grader.py [-h] [--test] [--log_converastions]
                 submissions_dir rubric_filename [thread_count] [results_dir]

positional arguments:
  submissions_dir      Directory containing submissions downloaded from Canvas
  rubric_filename      The json file containing the rubric for the assignment
  thread_count         The number of threads used to run the grader. Recommended for faster calculation.  
                       Will use a single thread if not specified.
  results_dir          The directory that the results will be created in.

options:
  -h, --help           show the help message and exit
  --test               Use this option to run only on the first 3 submission (used for testing purposes)  
  --log_converastions  Use this option to log all the LLM conversations. Will create one log file per     
                       submission.
```

The code then uses the criteria to analyze each submission. The result is a csv file containing one row per submission and one column for each criteria.
If the optional argument log-llm-converations is used, will also create an output log files per submission to show the conversations with the LLM for each of them.
