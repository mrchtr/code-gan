from utils.Preprocessor import clean_text

def decode(inp):
    """
    Decode given input into source code.
    :param inp: string sequence
    :return:
    """
    rgx_list = ["§"]
    return clean_text(inp, rgx_list, "\n")
