import requests
import argparse
from utility import TextUtil, PandasUtil, get_safe_filename, html_to_text, Confirm
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
    
    def add_options(self, *options: tuple[str, str]): # options is not dict since it can have multiple values for same key, e.g. include
        url = self.value
        if '?' in url:
            url += '&'
        elif len(options) > 0:
            url += '?'
        url += '&'.join(f"{k}={v}" for k, v in options)
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

def GET_url(url: URL, *options: tuple[str, str], quiet:bool=False):
    headers = {
        'Authorization': f'Bearer {CANVAS_ACCESS_TOKEN}',
        'Content-Type': 'application/json'
    }
    url = url.add_options(*options)
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
        
def fetch_list(url: URL, max_size: int|None, *options: tuple[str, str]):
    items = []
    if max_size is not None and not any([k == "per_page" for k, _ in options]):
        options = (*options, ("per_page", str(max_size)))
    while max_size is None or len(items) < max_size:
        # attempt to fetch all at once, otherwise it will fetch 10 at a time by default which is slower
        headers, this_items = GET_url(url, *options)
        if this_items is None:
            break
        items += this_items
        links = headers.get("link", "").split(',')
        links = dict([(link.split(';')[1].split('"')[1], link.split(';')[0][1:-1]) for link in links])
        if 'next' not in links or ('last' in links and links['current'] == links['last']):
            break
        url = URL(links['next'])
        options = () # the next link is comming from next, and it should already include all the required options. We should not manually change it.
    if max_size is not None:
        items = items[:max_size]
    return items

def get_course_info(course_id:int, should_print:bool=False):
    _, course_info = GET_url(COURSE_URL.to_url(course_id=course_id))
    assert course_info is not None, f"Course {course_id} does not exist."
    if should_print:
        print(TextUtil.get_colored_text(course_info['name'], TextUtil.TEXT_COLOR.Green))
    return course_info

def get_assignment_info(course_id:int, assignment_id:int, should_print:bool=False):
    _, assignment_info = GET_url(ASSIGNMENT_URL.to_url(course_id=course_id, assignment_id=assignment_id))
    assert assignment_info is not None, f"Assignment {assignment_id} does not exist."
    if should_print:
        print(TextUtil.get_colored_text(assignment_info['name'], TextUtil.TEXT_COLOR.Green))
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
        get_course_info(args.course_id, True)
        
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
        get_assignment_info(args.course_id, args.assignment_id, True)

class ListSubmissions(ListFromAssignment):
    @staticmethod
    def get_parse_info() -> dict:
        return {"name": "list_submissions", "help": "List all submissions of a assignment_id from a course_id"}
    @staticmethod
    def execute(args: argparse.Namespace):
        super(ListSubmissions, ListSubmissions).execute(args)
        super(ListSubmissions, ListSubmissions).print_list(args, SUBMISSIONS_URL.to_url(course_id=args.course_id, assignment_id=args.assignment_id),
                ["id", "user_id", "workflow_state", "grade", "attempt"],
                [None, 25, 15, 15, 15]
                )
        
class DownloadSubmissions(Command):
    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser.add_argument("course_id", help="The identifier of the course. Can be retrieved using list_courses.")
        parser.add_argument("assignment_id", help="The identifier of the assignment. Can be retrieved using list_assignments.")
        parser.add_argument('--max_count', type=int, required=False, default=MAX_FETCH, help='Maximum number of courses to fetch')
        parser.add_argument('--output_dirname', type=str, required=False, default="submissions", help='Output directory for downloading submissions')
        parser.add_argument('--all_attempts', action='store_true', help="Include all attempts (not just latest) for each submission. All comments are included whether or not this option is chosen.")
    @staticmethod
    def get_parse_info() -> dict:
        return {"name": "download_submissions", "help": "Download submissions for a given assignment"}
    @staticmethod
    def _get_dirs(parent_dirname, course_info, assignment_info):
        if not os.path.exists(args.output_dirname):
            os.makedirs(args.output_dirname)
        name = get_safe_filename(f"{course_info['name']}_{assignment_info['name']}", True)
        csv_filename = os.path.join(parent_dirname, get_safe_filename(name, extension="csv"))
        download_dir = os.path.join(parent_dirname, name)
        return csv_filename, download_dir
    @staticmethod
    def _get_quiz_questions(assignment_info):
        questions=[]
        if assignment_info.get("quiz_id", None) is not None:
            _, questions = GET_url(QUIZ_QUESTIONS_URL.to_url(course_id=args.course_id, quiz_id=assignment_info["quiz_id"]))
            assert questions is not None, f"Questions not found for quiz {assignment_info['quiz_id']}"
        return questions
    @staticmethod
    def _get_sortable_name(student):
        sortable_name = "_".join([part.strip() for part in student["user"]["sortable_name"].split(",")])
        if sortable_name == "Student_Test":
            # possible: ΩΩΩ_Test_Student (but I have to fix safe_filename, not that test student gonna have any submission though)
            # possible: ~Test_Student show latest in the spreadsheet, but earliest in files download (not that test student gonna have any submission though))
            sortable_name = "zzz_Student_Test" # TODO fix this. This is not a good way to handle this.
        return sortable_name
    @staticmethod
    def _get_attempt_info(attempt, student, questions, submission):
        downloads = []
        submission_type = attempt["submission_type"]
        sortable_name = DownloadSubmissions._get_sortable_name(student)
        attempt_info: dict[str, str] = {}
        if submission_type is None: # for some reason, no submission is considered a history for submission
            return {"Empty": True}, [] # kept for comments
        if submission_type == "online_url":
            attempt_info = {"submission": attempt["url"]}
        elif submission_type == "online_quiz":
            for i, question in enumerate(questions):
                attempt_info[question["question_name"]] = attempt["submission_data"][i]["text"]
        elif submission_type == "online_upload":
            attachments = attempt["attachments"]
            for i, attachment in enumerate(attachments):
                url = attachment["url"]
                attachment_filename = get_safe_filename(f"{sortable_name}_{i}_{student['user']['id']}") # id included since students can have the same name
                attempt_info[f"attachment[{i}]"] = attachment_filename
                downloads.append({"filename": attachment_filename, "url": url})
        else:
            TextUtil.warn(f"Unsupported submission type: {submission_type} for submission {submission['id']}")
        return attempt_info, downloads
    @staticmethod
    def _write_to_file(submission_data, csv_filename):
        submission_data.sort(key=lambda row: row.get("sortable_name", ""))
        for row in submission_data:
            for key, value in row.items():
                if isinstance(value, str):
                    row[key] = html_to_text(value)
        col_order = [ # TODO enum these? shouldn't repeat
            "course_id", "assignment_id", "submission_id", "student_id", "student_name", "student_email",
            "sortable_name", "attempt_number"]
        df = pd.DataFrame(submission_data)
        col_order += [col for col in df.columns if col not in col_order]
        df = df[col_order]
        df.to_csv(csv_filename, index=False)
        print(TextUtil.get_colored_text(f"Saved as {csv_filename}", TextUtil.TEXT_COLOR.Red))
    @staticmethod
    def _download_attachments(downloads, download_dir):
        if len(downloads) > 0:
            if not os.path.exists(download_dir):
                os.makedirs(download_dir)
            print(TextUtil.get_colored_text(f"Donwloading attachments at {download_dir}", TextUtil.TEXT_COLOR.Blue))
            Parallel(n_jobs=50)(delayed(GET_download)(URL(downloads[i]["url"]), os.path.join(download_dir, downloads[i]["filename"]), True) for i in tqdm(range(len(downloads))))
    @staticmethod
    def _summarize_rubric(rubric) -> str:
        ratings_summary = '\n'.join(
            f'[{rating["points"]}] ' +\
            f'{rating["description"]}' +\
            (f'({rating["long_description"]})' if len(rating["long_description"]) > 0 else '')
            for rating in rubric["ratings"])
        return f'[{rubric["points"]}] point(s)\n' +\
               f'desc: {rubric["description"]}\n'+\
               (f'long desc: {rubric["long_description"]}\n' if len(rubric["long_description"]) > 0 else '') +\
               ratings_summary
    @staticmethod
    def execute(args: argparse.Namespace):
        course_info = get_course_info(args.course_id, True)
        assignment_info = get_assignment_info(args.course_id, args.assignment_id, True)
        questions=DownloadSubmissions._get_quiz_questions(assignment_info)
        csv_filename, download_dir = DownloadSubmissions._get_dirs(args.output_dirname, course_info, assignment_info)
        students = fetch_list(ENROLLMENTS_URL.to_url(course_id=args.course_id), MAX_FETCH)
        submissions = fetch_list(
            SUBMISSIONS_URL.to_url(course_id=args.course_id, assignment_id=args.assignment_id),
            args.max_count,
            ("include[]", "submission_history"),
            ("include[]", "submission_comments"),
            ("include[]", "rubric_assessment"),
        )
        submission_data = []
        downloads=[]
        submission_data.append(dict([("student_name", "Reference/MaxGrades")] +
                                    [("grade", assignment_info["points_possible"])] +
                                    [("new_grade", assignment_info["points_possible"])] +
                                    [("new_comment", "Placeholder for new comment")] +
                                    [(f'rubric[{i}]:{rubric["id"]}g', f'{rubric["points"]}') for i, rubric in enumerate(assignment_info.get("rubric", None) or [])] +
                                    [(f'rubric[{i}]:{rubric["id"]}c', DownloadSubmissions._summarize_rubric(rubric)) for i, rubric in enumerate(assignment_info.get("rubric", None) or [])] +
                                    [(f'new_rubric[{i}]:g', f'{rubric["points"]}') for i, rubric in enumerate(assignment_info.get("rubric", None) or [])] +
                                    [(f'new_rubric[{i}]:c', DownloadSubmissions._summarize_rubric(rubric)) for i, rubric in enumerate(assignment_info.get("rubric", None) or [])] +
                                    [(question["question_name"],question["question_text"]) for question in questions]
                                ))
        for submission in submissions:
            student = next((s for s in students if s["user"]["id"] == submission["user_id"]), None)
            if student is None:
                TextUtil.warn(f"Submission found for student {submission['user_id']}, who is not enrolled in this class - skipped")
                continue
            sortable_name = DownloadSubmissions._get_sortable_name(student)
            history = submission["submission_history"]
            history = history[-1:] if not args.all_attempts else history
            for attempt_index, attempt in enumerate(history):
                comments = dict([(f"comment[{index}]", f"attempt[{comment['attempt']}] {comment['author_name']}\n{comment['comment']}")
                                 for index, comment in enumerate(submission["submission_comments"])
                                 if comment["attempt"] == attempt_index or comment["attempt"] is None or not args.all_attempts])
                rubric = dict([(f'rubric[{i}]:{id}g', f'{assessment.get("points", None)}') for i, (id, assessment) in enumerate((submission.get("rubric_assessment", None) or {}).items())] +
                              [(f'rubric[{i}]:{id}c', f'{assessment.get("comments", None)}') for i, (id, assessment) in enumerate((submission.get("rubric_assessment", None) or {}).items())])
                attempt_info, attachments = DownloadSubmissions._get_attempt_info(attempt, student, questions, submission)
                downloads += attachments
                submission_data.append({
                    "course_id": args.course_id,
                    "assignment_id": args.assignment_id,
                    "submission_id": submission["id"],
                    "student_id": student["user"]["id"],
                    "student_name": student["user"]["name"],
                    "student_email": student["user"]["login_id"],
                    "sortable_name": sortable_name,
                    "attempt_number": attempt_index+1,
                    "grade": submission["grade"],
                    **attempt_info,
                    **comments,
                    **rubric,
                })
        DownloadSubmissions._write_to_file(submission_data, csv_filename)
        DownloadSubmissions._download_attachments(downloads, download_dir)

class GradeSubmissions(Command):
    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser.add_argument("grade_filename", help="The file including grades for the student. The file should follow format similar to output of download_submissions. Specifically, each row should have course_id, assignment_id and student_id, or it will be skipped. student_name will be used for logging purposes if provided.")
        parser.add_argument('--grade_col', type=str, required=True, help="The column associated with grade. Note that this value will be ignored if instead a rubric is used.")
        parser.add_argument('--comment_col', type=str, required=True, help="The column associated with comment.")
    @staticmethod
    def get_parse_info() -> dict:
        return {"name": "grade_submissions", "help": "Upload submission grades for a given assignment"}
    @staticmethod
    def execute(args: argparse.Namespace):
        conf = Confirm()
        df = pd.read_csv(args.grade_filename)
        course_id = PandasUtil.get_if_all_same(df, "course_id")
        assignment_id = PandasUtil.get_if_all_same(df, "assignment_id")
        assignment_info = get_assignment_info(course_id, assignment_id, True)
        rubric_grade_col_names = {}
        rubric_comment_col_names = {}
        if assignment_info.get("use_rubric_for_grading", False):
            if Confirm().ask("The grade for this assignment can be based on the rubric. Would you like to use rubric columns for grading?\n" +\
                             TextUtil.get_colored_text(f"NOTE: THIS ACTION FORCES TO IGNORE THE PROVIDED GRADE_COL, \"{args.grade_col}\"", TextUtil.TEXT_COLOR.Red)):
                for i, rubric in enumerate(assignment_info["rubric"] or []):
                    rubric_grade_col_names[rubric["id"]] = PandasUtil.ask_col_name(df, f'rubric[{i}]:{rubric["id"]}\n{DownloadSubmissions._summarize_rubric(rubric)}\nWhich column refers to the ' + TextUtil.get_colored_text('GRADE', TextUtil.TEXT_COLOR.Yellow) + ' of this criterion?')
                    rubric_comment_col_names[rubric["id"]] = PandasUtil.ask_col_name(df, f'rubric[{i}]:{rubric["id"]}\n{DownloadSubmissions._summarize_rubric(rubric)}\nWhich column refers to the ' + TextUtil.get_colored_text('COMMENT', TextUtil.TEXT_COLOR.Yellow) + 'on this criterion?')
        for _, row in df.iterrows():
            student_id, student_name, student_email, grade, comment = PandasUtil.multi_get(row,
                "student_id", "student_name", "student_email", args.grade_col, args.comment_col)
            if student_id is None or not TextUtil.is_type(student_id, int, f"Not a valid integer for student_id: {student_id} - skipped"):
                continue
            student_id = int(student_id)
            params = {}
            grade_text:str = "<None>"
            comment_text:str = "<None>"
            if comment is not None:
                params["comment[text_comment]"] = comment
                comment_text = f"\"{comment}\""
            if len(rubric_grade_col_names) > 0:
                grade_text = "rubric"
                comment_text += "_rubric"
                for rubric in (assignment_info["rubric"] or []):
                    rubric_id = rubric["id"]
                    grade_col_name = rubric_grade_col_names[rubric_id]
                    comment_col_name = rubric_comment_col_names[rubric_id]
                    rubric_grade = PandasUtil.get(row, grade_col_name)
                    rubric_comment = PandasUtil.get(row, comment_col_name)
                    if rubric_grade is not None and TextUtil.is_type(rubric_grade, float, f"Rubric grade for {grade_col_name} is not a valid float: {rubric_grade}, for student_id: {student_id} - skipped"):
                        params[f"rubric_assessment[{rubric_id}][points]"] = f"{rubric_grade}"
                    grade_text += f"_{rubric_grade}"
                    if rubric_comment is not None:
                        params[f"rubric_assessment[{rubric_id}][comments]"] = f"{rubric_comment}"
                    comment_text += f"_\"{rubric_comment}\""
            elif grade is not None and TextUtil.is_type(grade, float, f"Grade is not a valid float: {grade}, for student_id: {student_id} - skipped"):
                params["submission[posted_grade]"] = str(grade)
                grade_text = str(grade)
            if len(params) > 0 and conf.ask(f"Applying grade for {student_name}({student_email}) to {grade_text} and putting comment:\n{comment_text}", no_all=False):
                PUT_url(USER_SUBMISSION_URL.to_url(course_id=course_id, assignment_id=assignment_id,
                                                   user_id=student_id), params=params)

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
