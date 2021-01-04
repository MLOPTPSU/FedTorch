# -*- coding: utf-8 -*-
import os
import shutil


def get_current_path(args, rank):
    paths = args.resume.split(',')
    splited_paths = map(
        lambda p: p.split('/')[-1].split('-')[: 1], paths)
    splited_paths_dict = dict([
        (path, paths[ind]) for ind, path in enumerate(splited_paths)])
    return splited_paths_dict[str(rank)]


def build_dir(path, force):
    """build directory."""
    if os.path.exists(path) and force:
        shutil.rmtree(path)
        os.mkdir(path)
    elif not os.path.exists(path):
        os.mkdir(path)
    return path


def build_dirs(path):
    try:
        os.makedirs(path)
    except Exception as e:
        print(' encounter error: {}'.format(e))


def remove_folder(path):
    try:
        shutil.rmtree(path)
    except Exception as e:
        print(' encounter error: {}'.format(e))


def list_files(root_path):
    dirs = os.listdir(root_path)
    return [os.path.join(root_path, path) for path in dirs]
