import re

def extract_json_from_text(text: str):
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group()
    return None

def is_template_string(s: str) -> bool:
    try:
        s.format()
        return False
    except KeyError:
        return True