import requests
import argparse
from utility import TextUtil, warn, get_safe_filename, html_to_text
from enum import StrEnum
from auth import CANVAS_ACCESS_TOKEN
import pandas as pd
import os
from joblib import delayed, Parallel
from tqdm import tqdm

MAX_FETCH = 500
DOWNLOAD_THREAD_COUNT = 50

class URL:
    def __init__(self, value: str):
        self.value = value

    def add(self, addition):
        if addition[0] != '/':
            addition = '/' + addition
        return URL(self.value + addition)
    
    def parametrize(self, parameter_name:str, custom_syntax:str|None=None):
        return ParametrizedURL(self).parametrize(parameter_name, custom_syntax)
    
    def add_options(self, options):
        url = self.value
        if '?' in url:
            url += '&'
        elif len(options) > 0:
            url += '?'
        url += '&'.join(f"{k}={v}" for k, v in options.items())
        return URL(url)
    
    def __repr__(self) -> str:
        return self.value
    
class ParametrizedURL:
    def __init__(self, url: URL, parameters: set[str]=set()):
        self.url = url
        self.parameters = parameters

    @staticmethod
    def key(parameter_name):
        return f"{{{parameter_name}}}"
    
    def parametrize(self, parameter_name:str, custom_syntax:str|None=None):
        assert parameter_name not in self.parameters, f"url already parametrized on {parameter_name}"
        key = ParametrizedURL.key(parameter_name)
        if custom_syntax is None:
            custom_syntax = key
        assert key in custom_syntax, f"custom syntax does not include placeholder for {parameter_name}"
        assert key not in repr(self.url), f"url already includes a parameter placeholder for {parameter_name}"
        return ParametrizedURL(self.url.add(custom_syntax), set([parameter_name] + list(self.parameters)))

    def add(self, addition: str):
        return ParametrizedURL(self.url.add(addition), self.parameters)
    
    def to_url(self, **kwargs) -> URL:
        assert set(kwargs.keys()) == self.parameters, ", ".join(list(set(kwargs.keys()))) + " != " + ", ".join(self.parameters)
        url = repr(self.url)
        for parameter_name in self.parameters:
            key = ParametrizedURL.key(parameter_name)
            url = url.replace(key, str(kwargs[parameter_name]))
        return URL(url)
    
CANVAS_URL = URL('https://canvas.ucsc.edu')
BASE_URL = CANVAS_URL.add('api/v1')
COURSES_URL = BASE_URL.add('courses')
COURSE_URL = COURSES_URL.parametrize("course_id")
ENROLLMENTS_URL = COURSE_URL.add("enrollments")
ASSIGNMENTS_URL = COURSE_URL.add("assignments")
ASSIGNMENT_URL = ASSIGNMENTS_URL.parametrize("assignment_id")
SUBMISSIONS_URL = ASSIGNMENT_URL.add("submissions")
USER_SUBMISSION_URL = SUBMISSIONS_URL.parametrize("user_id")
QUIZZES_URL = COURSE_URL.add("quizzes")
QUIZ_URL = QUIZZES_URL.parametrize("quiz_id")
QUIZ_QUESTIONS_URL = QUIZ_URL.add("questions")

def GET_url(url: URL, options: dict = {}, quiet:bool=False):
    headers = {
        'Authorization': f'Bearer {CANVAS_ACCESS_TOKEN}',
        'Content-Type': 'application/json'
    }
    url = url.add_options(options)
    if not quiet:
        print(TextUtil.get_colored_text(f"GET {url}", TextUtil.TEXT_COLOR.Yellow))
    response = requests.get(repr(url), headers=headers)

    if response.status_code == 200:
        return response.headers, response.json()  # Return the JSON response if successful
    else:
        print(f"Error: {response.status_code} for GET {repr(url)}")

        return response.headers, None
    
def GET_download(url: URL, filename: str, quiet:bool=False, use_originial_extension:bool=True):
    headers = {
        'Authorization': f'Bearer {CANVAS_ACCESS_TOKEN}',
        'Content-Type': 'application/json'
    }
    if not quiet:
        print(TextUtil.get_colored_text(f"GET download {url}", TextUtil.TEXT_COLOR.Yellow))
    response = requests.get(repr(url), headers=headers)
    extension = None
    for property in response.headers["Content-Disposition"].split(";"):
        if property.split("=")[0].strip() == "filename":
            extension = get_safe_filename(property.split(".")[-1]) # can have " at the end, fixed by safe_filename
    if use_originial_extension and extension is not None:
        filename = f"{filename}.{extension}"
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024): 
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)
        return response.headers, None
    else:
        print(f"Error: {response.status_code} for GET {repr(url)}")
        return response.headers, None

def PUT_url(url: URL, params: dict[str, str] = {}, quiet:bool=False):
    headers = {
        'Authorization': f'Bearer {CANVAS_ACCESS_TOKEN}',
        'Content-Type': 'application/json'
    }
    if not quiet:
        print(TextUtil.get_colored_text(f"PUT {url}", TextUtil.TEXT_COLOR.Yellow))
    response = requests.put(repr(url), params=params, headers=headers)

    if response.status_code == 200:
        return response.headers, response.json()  # Return the JSON response if successful
    else:
        print(f"Error: {response.status_code} for PUT {repr(url)}")
        return response.headers, None
        
def fetch_list(url: URL, max_size: int|None, options: dict = {}):
    items = []
    while max_size is None or len(items) < max_size:
        # attempt to fetch all at once, otherwise it will fetch 10 at a time by default which is slower
        headers, this_items = GET_url(url, options | ({"per_page": max_size} if max_size is not None else {}))
        if this_items is None:
            break
        items += this_items
        links = headers.get("link", "").split(',')
        links = dict([(link.split(';')[1].split('"')[1], link.split(';')[0][1:-1]) for link in links])
        if 'next' not in links or ('last' in links and links['current'] == links['last']):
            break
        url = URL(links['next'])
    if max_size is not None:
        items = items[:max_size]
    return items

def get_course_info(course_id):
    _, course_info = GET_url(COURSE_URL.to_url(course_id=course_id))
    assert course_info is not None, f"Course {course_id} does not exist."
    return course_info

def get_assignment_info(course_id, assignment_id):
    _, assignment_info = GET_url(ASSIGNMENT_URL.to_url(course_id=course_id, assignment_id=assignment_id))
    assert assignment_info is not None, f"Assignment {assignment_id} does not exist."
    return assignment_info

class Command:
    @staticmethod
    def execute(args: argparse.Namespace) -> None:
        raise Exception("not implemented yet")
    
    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        raise Exception("not implemented yet")
    
    @staticmethod
    def get_parse_info() -> dict:
        raise Exception("not implemented yet")
    
    @classmethod
    def subscribe(cls, subparsers) -> None:
        parser = subparsers.add_parser(**cls.get_parse_info())
        cls.add_args(parser)
        parser.set_defaults(func=cls.execute)

class ListCommand(Command):
    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser.add_argument('--max_count', type=int, required=False, default=MAX_FETCH, help='Maximum number of courses to fetch')
        parser.add_argument('--no_trunc', action='store_true', help="Use this option to prevent truncating the fields.")

    @staticmethod
    def extract(item: dict, attr: str, trunc:int|None=None) -> str:
        parts = attr.split(":")
        for part in parts:
            item = item.get(part, None)
            if item is None:
                return TextUtil.truncate("<None>", trunc)
        return TextUtil.truncate(repr(item), trunc)

    @staticmethod
    def print_list(args: argparse.Namespace, list_url: URL, attrs, truncates):
        items = fetch_list(list_url, args.max_count)
        rows = [[i] + [ListCommand.extract(item, attr, None if args.no_trunc else trunc) for attr, trunc in zip(attrs, truncates)] for i, item in enumerate(items)]
        TextUtil.pretty_print_list(["-"] + attrs, rows)

class ListCourses(ListCommand):
    @staticmethod
    def get_parse_info() -> dict:
        return {"name": "list_courses", "help": "List available courses for the owner of the token"}
    @staticmethod
    def execute(args: argparse.Namespace):
        super(ListCourses, ListCourses).print_list(args, COURSES_URL,
                ["id", "course_code", "name"],
                [None, 25, 40]
                )
    
class ListFromCourse(ListCommand):
    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        super(ListFromCourse, ListFromCourse).add_args(parser)
        parser.add_argument("course_id", help="The identifier of the course. Can be retrieved using list_courses.")
    @staticmethod
    def execute(args: argparse.Namespace):
        print(TextUtil.get_colored_text(get_course_info(args.course_id)['name'], TextUtil.TEXT_COLOR.Green))
        
class ListStudents(ListFromCourse):
    @staticmethod
    def get_parse_info() -> dict:
        return {"name": "list_students", "help": "List all students enrolled in course_id"}
    @staticmethod
    def execute(args: argparse.Namespace):
        super(ListStudents, ListStudents).execute(args)
        super(ListStudents, ListStudents).print_list(args, ENROLLMENTS_URL.to_url(course_id=args.course_id),
                ["user:id", "user:name", "user:login_id", "type"], #can also use user_id, which should be same as user:id
                [None, 30, 20, 20]
                )

class ListAssignments(ListFromCourse):
    @staticmethod
    def get_parse_info() -> dict:
        return {"name": "list_assignments", "help": "List all assignments of course_id"}
    @staticmethod
    def execute(args: argparse.Namespace):
        super(ListAssignments, ListAssignments).execute(args)
        super(ListAssignments, ListAssignments).print_list(args, ASSIGNMENTS_URL.to_url(course_id=args.course_id),
                ["id", "name", "workflow_state", "due_at", "submission_types"],
                [None, 20, 15, 15, 30]
                )
        
class ListFromAssignment(ListFromCourse):
    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        super(ListFromAssignment, ListFromAssignment).add_args(parser)
        parser.add_argument("assignment_id", help="The identifier of the assignment. Can be retrieved using list_assignments.")
    @staticmethod
    def execute(args: argparse.Namespace):
        super(ListFromAssignment, ListFromAssignment).execute(args)
        print(TextUtil.get_colored_text(get_assignment_info(args.course_id, args.assignment_id)['name'], TextUtil.TEXT_COLOR.Green))

class ListSubmissions(ListFromAssignment):
    @staticmethod
    def get_parse_info() -> dict:
        return {"name": "list_submissions", "help": "List all submissions of a assignment_id from a course_id"}
    @staticmethod
    def execute(args: argparse.Namespace):
        super(ListSubmissions, ListSubmissions).execute(args)
        super(ListSubmissions, ListSubmissions).print_list(args, SUBMISSIONS_URL.to_url(course_id=args.course_id, assignment_id=args.assignment_id),
                ["id", "name", "workflow_state", "due_at", "lock_at"],
                [None, 25, 15, 15, 15]
                )
        
class DownloadSubmissions(Command):
    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser.add_argument("course_id", help="The identifier of the course. Can be retrieved using list_courses.")
        parser.add_argument("assignment_id", help="The identifier of the assignment. Can be retrieved using list_assignments.")
        parser.add_argument('--max_count', type=int, required=False, default=MAX_FETCH, help='Maximum number of courses to fetch')
    @staticmethod
    def get_parse_info() -> dict:
        return {"name": "download_submissions", "help": "Download submissions for a given assignment"}
    @staticmethod
    def execute(args: argparse.Namespace):
        course_info = get_course_info(args.course_id)
        print(TextUtil.get_colored_text(course_info['name'], TextUtil.TEXT_COLOR.Green))
        assignment_info = get_assignment_info(args.course_id, args.assignment_id)
        print(TextUtil.get_colored_text(assignment_info['name'], TextUtil.TEXT_COLOR.Green))
        dirname = get_safe_filename(f"{course_info['name']}_{assignment_info['name']}", True)
        filename = get_safe_filename(dirname, extension="csv")
        students = fetch_list(ENROLLMENTS_URL.to_url(course_id=args.course_id), MAX_FETCH)
        submissions = fetch_list(
            SUBMISSIONS_URL.to_url(course_id=args.course_id, assignment_id=args.assignment_id),
            args.max_count,
            {"include[]": "submission_history"}
        )
        submission_data = []
        questions=[]
        downloads=[]
        if assignment_info.get("quiz_id", None) is not None:
            _, questions = GET_url(QUIZ_QUESTIONS_URL.to_url(course_id=args.course_id, quiz_id=assignment_info["quiz_id"]))
            assert questions is not None, f"Questions not found for quiz {assignment_info['quiz_id']}"
            submission_data.append(dict([(question["question_name"],question["question_text"]) for question in questions]))
        for submission in submissions:
            history = submission["submission_history"]
            attempts = len(history)
            student = next((s for s in students if s["user"]["id"] == submission["user_id"]), None)
            if student is None:
                warn(f"Submission found for student {submission['user_id']}, who is not enrolled in this class. (skipped in results)")
                continue
            latest_attempt = history[-1]
            submission_type = latest_attempt["submission_type"]
            sortable_name = "_".join([part.strip() for part in student["user"]["sortable_name"].split(",")])
            value: dict[str, str] = {}
            if submission_type is None: # for some reason, no submission is considered a history for submission
                attempts = 0
            elif submission_type == "online_url":
                value = {"submission": latest_attempt["url"]}
            elif submission_type == "online_quiz":
                for i, question in enumerate(questions):
                    value[question["question_name"]] = latest_attempt["submission_data"][i]["text"]
            elif submission_type == "online_upload":
                if not os.path.exists(dirname):
                    os.makedirs(dirname)
                attachments = latest_attempt["attachments"]
                for i, attachment in enumerate(attachments):
                    url = attachment["url"]
                    attachment_filename = get_safe_filename(f"{sortable_name}_{i}")
                    value[f"attachment[{i}]"] = attachment_filename
                    downloads.append({"filename": attachment_filename, "url": url})
            else:
                warn(f"Unsupported submission type: {submission_type} for submission {submission['id']}")
            submission_data.append({
                "course_id": args.course_id,
                "assignment_id": args.assignment_id,
                "submission_id": submission["id"],
                "student_id": submission["user_id"],
                "student_name": student["user"]["name"],
                "sortable_name": sortable_name,
                "student_email": student["user"]["login_id"],
                "attempts": attempts,
                "grade": submission["grade"],
                **value
            })
        submission_data.sort(key=lambda row: row["sortable_name"])
        for row in submission_data:
            for key, value in row.items():
                if isinstance(value, str):
                    row[key] = html_to_text(value)
        col_order = [
            "course_id", "assignment_id", "submission_id", "student_id", "student_name", "student_email",
            "sortable_name", "attempts"]
        df = pd.DataFrame(submission_data)
        col_order += [col for col in df.columns if col not in col_order]
        df = df[col_order]
        df.to_csv(filename, index=False)
        print(TextUtil.get_colored_text(f"Saved as {filename}", TextUtil.TEXT_COLOR.Red))
        if len(downloads) > 0:
            print(TextUtil.get_colored_text(f"Donwloading attachments at {dirname}", TextUtil.TEXT_COLOR.Blue))
            Parallel(n_jobs=50)(delayed(GET_download)(URL(downloads[i]["url"]), os.path.join(dirname, downloads[i]["filename"]), True) for i in tqdm(range(len(downloads))))

class GradeSubmissions(Command):
    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser.add_argument("grade_filename", help="The file including grades for the student. The file should follow format similar to output of download_submissions.")
        parser.add_argument('--grade_col', type=str, required=False, default="grade", help="The column associated with grade.")
        parser.add_argument('--comment_col', type=str, required=False, default="grade", help="The column associated with grade.")

    @staticmethod
    def get_parse_info() -> dict:
        return {"name": "grade_submissions", "help": "Download submissions for a given assignment"}
    @staticmethod
    def execute(args: argparse.Namespace):
        ask = True
        yes_no = None
        df = pd.read_csv(args.grade_filename)
        course_ids = set([int(row["course_id"]) for _, row in df.iterrows() if pd.notna(row["course_id"])])
        assert len(course_ids) == 1, "More than one or zero course_id found in the grading spreadsheet."
        course_id = list(course_ids)[0]
        assignment_ids = set([int(row["assignment_id"]) for _, row in df.iterrows() if pd.notna(row["assignment_id"])])
        assert len(assignment_ids) == 1, "More than one or zero assignment_id found in the grading spreadsheet."
        assignment_id = list(assignment_ids)[0]
        for _, row in df.iterrows():
            grade = row[args.grade_col]
            student_id = row["student_id"]
            if pd.isna(student_id) or pd.isna(grade):
                continue
            user_id = int(student_id)
            comment = row.get(args.comment_col, None)
            params = {"submission[posted_grade]": str(grade)}
            if comment is not None:
                params["comment[text_comment]"] = comment
            if row["student_name"] != "Test Student":
                continue
            if ask:
                yes_no = None
            while yes_no not in ['y', 'n', 'yall', 'nall']:
                yes_no = input(f"[y/n/yall/nall] setting grade for {row['student_name']}({row['student_email']}) to {grade} and putting comment: {comment}")
            if yes_no in ['nall', 'yall']:
                ask=False
            if yes_no in ['n', 'nall']:
                continue
            PUT_url(USER_SUBMISSION_URL.to_url(course_id=course_id, assignment_id=assignment_id, user_id=user_id),
                    params=params)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Main Command Line Interface")
    subparsers = parser.add_subparsers(dest='command')
    ListCourses.subscribe(subparsers)
    ListStudents.subscribe(subparsers)
    ListAssignments.subscribe(subparsers)
    ListSubmissions.subscribe(subparsers)
    DownloadSubmissions.subscribe(subparsers)
    GradeSubmissions.subscribe(subparsers)

    args = parser.parse_args()
    
    if args.command:
        args.func(args)
    else:
        parser.print_help()
