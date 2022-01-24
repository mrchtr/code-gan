"""
Downloading all raw files of the dataset. Afterwards the preprocessing will be applied on each file.
Additionally, a tokenizer can be trained on the dataset.
"""
import shutil

import git
import os
from tqdm import tqdm

from utils.FileUtils import create_dir_if_not_exists
from utils.Preprocessor import preprocess

debug = True
train_name = "./100k_train"
eval_name = "./50k_eval"
os.chdir("./data")

def get_data():
    """
    Download all files of the 150k dataset.
    """
    filename = "./150k_sources/github_repos.txt"

    # load all urls
    with open(filename) as f:
        content = f.readlines()

    urls = [x.split("\t")[1][:-1] for x in content]
    print(f"Start crawling of {len(urls)} repositories ...")

    # clone all repositories
    i = 0
    for line in tqdm(content):
        if debug and i > 20:
            break
        hash, url = line.split("\t")
        url = url[:-1]
        i += 1
        try:
            git.Git("./").clone(url)
            git.Git(f"./{url.split('/')[-1]}").checkout(hash)
        except:
            print(f"Could not find repository {url}")
    i = 0

    # create dir if no exists
    try:
        os.mkdir(f"./{train_name}")
        os.mkdir(f"./{eval_name}")
    except:
        print(f"Dir already exsits")

    # load train files
    with open("./150k_sources/python100k_train.txt") as f:
        train_paths = f.readlines()

    not_found = []
    found = []
    for i in tqdm(range(len(train_paths))):
        path = train_paths[i].split("/")
        path = "/".join(path[2:len(path)])[:-1]
        filename = path.split("/")[-1]
        try:
            os.rename(path, f"./{train_name}/{i}_{filename}")
            found.append(path)
        except:
            # print(f"Fail to load: {path}")
            not_found.append(path)

    print(f"founded:  {len(found)}")
    print(f"not founded: {len(not_found)}")

    # load test files
    with open("./150k_sources/python50k_eval.txt") as f:
        train_paths = f.readlines()

    found_2 = []
    not_found_2 = []
    for i in tqdm(range(len(train_paths))):
        path = train_paths[i].split("/")
        path = "/".join(path[2:len(path)])[:-1]
        filename = path.split("/")[-1]
        try:
            os.rename(path, f"./{eval_name}/{i}_{filename}")
            found_2.append(path)
        except:
            not_found_2.append(path)

    print(f"founded: {found_2}")
    print(f"not founded: {not_found_2}")


def processing_dir(root, name="train", out_dir="dataset"):
    print(f"Start preprocessing files in dir: {root}")
    paths = []
    for root, dirnames, filenames in os.walk(root):
          for filename in filenames:
              if filename.endswith(('.py')):
                  paths.append((os.path.join(root, filename)))

    create_dir_if_not_exists(out_dir)
    out_filename = f"{out_dir}/out_{name}.txt"
    output_data = open(out_filename, "a")  # append mode
    print(f"Start processing {len(paths)} files")
    errors = 0
    i = 0
    for x in tqdm(paths):
      with open(x) as f:
        try:
          content = "".join(f.readlines())
          content = preprocess(content)
          output_data.write(content)
        except:
          errors = errors + 1
      i = i + 1

    if errors !=0:
        print(f"Can't processing {errors} files.")


def cleanup():
    for path, dirs, files in os.walk('./'):
        if "150k_sources" in dirs:
            dirs.remove("150k_sources")
            dirs.remove("100k_train")
            dirs.remove("50k_eval")
            dirs.remove("dataset")

        for name in dirs:
            shutil.rmtree(name)

if __name__ == '__main__':
    get_data()
    processing_dir(f"./{train_name}", "train")
    processing_dir(f"./{eval_name}", "test")
    cleanup()

