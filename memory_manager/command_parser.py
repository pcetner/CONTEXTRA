from typing import Tuple, Optional

def parse_user_input(user_input: str) -> Tuple[str, Optional[str]]:
    """Parse user input for commands and regular text"""
    if user_input.startswith('!important '):
        return 'important', user_input[10:].strip()
    elif user_input.startswith('!forget '):
        return 'forget', user_input[8:].strip()
    elif user_input.startswith('!summary'):
        return 'summary', None
    else:
        return 'message', user_input 