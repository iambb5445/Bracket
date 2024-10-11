from enum import StrEnum

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

def get_safe_filename(filename: str):
    keepcharacters = (' ','.','_', '-')
    return "".join(c for c in filename if c.isalnum() or c in keepcharacters).rstrip()