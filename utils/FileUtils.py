import os
import glob

def get_all_files(dir):
    """
    Get all files of the give directory
    :param dir: dir where the files are located
    :return: list of files
    """
    list_of_files = list()
    for (dir_path, dir_names, filenames) in os.walk(dir):
        list_of_files += [os.path.join(dir_path, file) for file in filenames]

    return list_of_files

def create_dir_if_not_exists(dir):
    """
    Create dir if not exists
    :param dir: dir path
    :return: sucessfully created
    """
    if not os.path.exists(dir):
        os.makedirs(dir)
