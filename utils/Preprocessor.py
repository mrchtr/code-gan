import re


def preprocess(inp):
    """
    Combines different preprocessing methods and return the preprocced string.
    :param inp: input string.
    :return: preprocessed string
    """
    inp = remove_comments(inp)
    inp = replace_tabs(inp)
    inp = replace_line_breaks(inp)
    return inp


def remove_comments(inp):
    """
    Remove comments on the given input
    :param inp: input string.
    :return: input string without comments
    """
    rgx_list = ["#.*\n", "\"""(.|[\r\n])*\""""]
    return clean_text(rgx_list, inp)

def replace_tabs(inp):
    """
    Tabs will be replaced by a special token.
    :param inp:
    :return:
    """

    rgx_list = ["^\s\s\s\s", "    "]
    return clean_text(rgx_list, inp, "<TAB>")

def replace_line_breaks(inp):
    """
    Tabs will be replaced by a special token.
    :param inp:
    :return:
    """

    rgx_list = ["\n"]
    return clean_text(rgx_list, inp, "<LB>")


def clean_text(rgx_list, text, replacement=''):
    """
    Remove parts that matches the regex list
    :param rgx_list: list of expressions
    :param text: given inp
    :return: cleaned inp
    """
    new_text = text
    for rgx_match in rgx_list:
        new_text = re.sub(rgx_match, replacement, new_text)
    return new_text

