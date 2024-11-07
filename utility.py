from enum import StrEnum
from tabulate import tabulate
import time
from bs4 import BeautifulSoup

class TextUtil:
    class TEXT_COLOR(StrEnum):
        Red = '\033[91m'
        Green = '\033[92m'
        Blue = '\033[94m'
        Cyan = '\033[96m'
        White = '\033[97m'
        Yellow = '\033[93m'
        Magenta = '\033[95m'
        Grey = '\033[90m'
        Black = '\033[90m'
        Default = '\033[99m'
        _Reset = '\033[0m'
    @staticmethod
    def dedent(s):
        # textwrap dedent is not working with this. It uses the first line to understand the number of indents.
        # import textwrap
        # return textwrap.dedent(s)
        return '\n'.join([line.lstrip() for line in s.split('\n')])

    @staticmethod
    def get_colored_text(text: str, color: TEXT_COLOR):
        return color + text + TextUtil.TEXT_COLOR._Reset
    
    @staticmethod
    def set_print_color(color: TEXT_COLOR|None):
        if color is None:
            color = TextUtil.TEXT_COLOR._Reset
        print(color, end='')

    @staticmethod
    def pretty_print_list(headers: list, rows: list[list]):
        for row in rows:
            if len(row) != len(headers):
                print("Warning: pretty printing the list has failed. Length of rows do not match the headers.")
                print(rows)
                return
        print(tabulate(rows, headers, tablefmt="orgtbl"))
    
    @staticmethod
    def truncate(text: str, max_size: int|None):
        if max_size is None or len(text) < max_size:
            return text
        if max_size < 3:
            return text[:max_size]
        return text[:max_size - 3] + "..."
        

def get_safe_filename(filename: str, timed:bool=False, extension:str|None=None):
    keepcharacters = (' ','.','_', '-')
    filename = "".join(c for c in filename if c.isalnum() or c in keepcharacters).rstrip()
    if timed:
        filename += f"_{int(time.time())}"
    if extension is not None:
        extension = "".join(c for c in extension if c.isalnum() or c in keepcharacters).rstrip()
        filename += f".{extension}"
    return filename

def warn(warning_message: str):
    print(TextUtil.get_colored_text(f"[WARNING]{warning_message}", TextUtil.TEXT_COLOR.Red))

def html_to_text(value, raw:bool=False):
    soup = BeautifulSoup(value, "html.parser")

    if raw:
        return soup.get_text()
    
    # Replace <li> with bullet points
    for li_tag in soup.find_all("li"):
        li_tag.insert_before("• ")
        li_tag.unwrap() 
    
    # Replace <b> with bold simulation (use **bold** for example)
    for b_tag in soup.find_all("b"):
        b_tag.insert_before("**")
        b_tag.insert_after("**")
        b_tag.unwrap()
    
    # Replace <i> with italics simulation (use /italic/ for example)
    for i_tag in soup.find_all("i"):
        i_tag.insert_before("/")
        i_tag.insert_after("/")
        i_tag.unwrap()
    
    # Replace <br> with a newline character
    for br_tag in soup.find_all("br"):
        br_tag.insert_before("\n")
        br_tag.unwrap()
    
    return soup.get_text()