import re
from tokenize import tokenize, untokenize, NUMBER, STRING, INDENT, DEDENT, COMMENT
from io import BytesIO

COMMENT_TOKEN = "<COMMENT>"
INT_TOKEN = "<INT_LIST>"
STR_TOKEN = "<STR_LIT>"
INDENT_TOKEN = "<INDENT>"
DEDENT_TOKEN = "<DEDENT>"
BOF_TOKEN = "<BOF>"
EOF_TOKEN = "<EOF>"


def preprocess(inp):
    """
    Combines different preprocessing methods and return the preproccded string.
    :param inp: input string.
    :return: preprocessed string
    """

    try:
        inp = replace_literals(inp)
    except:
        return ""
    inp = untokenize(inp).decode('utf-8')
    inp = replace_whitespace_not_needed(inp)
    inp = unquote_special_tokens(inp)
    inp = f"{BOF_TOKEN} {inp} {EOF_TOKEN}"
    inp = remove_not_needed_tokens(inp)
    inp = remove_empty_lines(inp)
    return inp


def replace_whitespace_not_needed(inp):
    rgx_list = ["\s(=)", "\s(\()", "\s(\))", "\s(\.)", "\s(,)"]
    return clean_text(rgx_list, inp, r"\1")


def unquote_special_tokens(inp):
    rgx_list = [f"'({INT_TOKEN})'", f"'({INDENT_TOKEN})'", f"'({DEDENT_TOKEN})'"]
    return clean_text(rgx_list, inp, r"\1")

def remove_not_needed_tokens(inp):
    rgx_list = [f"'({COMMENT_TOKEN})'"]
    return clean_text(rgx_list, inp, "")

def remove_empty_lines(inp):
    inp = inp.split("\n")
    inp = [x for x in inp if x.strip() != ""]
    return "\n".join(inp)

def clean_text(rgx_list, text, replacement=''):
    """
    Remove parts that matches the regex list
    :param rgx_list: list of expressions
    :param text: given inp
    :return: cleaned inp
    """
    new_text = text
    for rgx_match in rgx_list:
        new_text = re.sub(rgx_match, str(replacement), str(new_text))
    return new_text


def replace_literals(inp):
    """
    Replacing pyhton literals with special tokens.
    :param inp:
    :return:
    """
    g = tokenize(BytesIO(inp.encode('utf-8')).readline)
    result = []
    for toknum, tokval, _, _, _ in g:
        if toknum == NUMBER:  # replace NUMBER tokens with <INT_LIT>
            result.extend([
                (STRING, repr(INT_TOKEN))
            ])
        elif toknum == STRING:
            result.extend([
                (STRING, repr(STR_TOKEN))
            ])

        elif toknum == COMMENT:
            result.extend([
                (STRING, repr(COMMENT_TOKEN))
            ])

        elif toknum == INDENT:

            result.extend([
                (STRING, repr(INDENT_TOKEN))
            ])
        elif toknum == DEDENT:
            result.extend([
                (STRING, repr(DEDENT_TOKEN))
            ])
        else:
            result.append((toknum, tokval))

    return result
