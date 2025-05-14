"""This module provides helper functions for the simulation module."""

import os
from time import sleep
from fcntl import flock, LOCK_EX, LOCK_UN
from pathlib import Path


def add_simulation_task_to_queue2(task_name: str | list, cache_path: str = None):
    """
    Add a simulation task to the queue.
    :param task_name: Filename of the simulation task to add to the queue.
        The file should be a '.tpr' file, but without the extension.
        A list of files can also be provided, in which case all files are added to the queue.
    :param cache_path: Path to the cache directory. This path is used to store the task queue.
        If not provided, the 'CHESPEX_CACHE' environment variable is used, or the default
        '~/.chespex/cache' directory is used.
    """
    if task_name is None:
        return
    if not isinstance(task_name, list):
        task_name = [task_name]
    if cache_path is not None:
        cache_path = Path(cache_path)
    elif "CHESPEX_CACHE" in os.environ:
        cache_path = Path(os.environ["CHESPEX_CACHE"])
    else:
        cache_path = Path("~/.chespex/cache").expanduser()
    queue_filename = cache_path / "task_queue.txt"
    with open(queue_filename, "a", encoding="utf-8") as queue_list_file:
        flock(queue_list_file, LOCK_EX)
        for task in task_name:
            try:
                task = Path(task).resolve()
                # print(f"Adding task {task} to queue")
                queue_list_file.write(str(task) + "\n")
            except ValueError:
                print(f"ValueError: {task} is not a valid path")
        flock(queue_list_file, LOCK_UN)


def _wait_for_lock_file(lock_file: str):
    """
    Wait until the lock file is removed.
    :param lock_file: Path to the lock file.
    """
    while True:
        if not os.path.exists(lock_file):
            try:
                file = open(lock_file, "xb")
                file.close()
                break
            except FileExistsError:
                pass
        sleep(0.1)


def _remove_lock_file(lock_file: str):
    """
    Remove the lock file.
    :param lock_file: Path to the lock file.
    """
    os.remove(lock_file)


def add_simulation_task_to_queue(task_name: str | list, cache_path: str = None):
    """
    Add a simulation task to the queue.
    :param task_name: Filename of the simulation task to add to the queue.
        The file should be a '.tpr' file, but without the extension.
        A list of files can also be provided, in which case all files are added to the queue.
    :param cache_path: Path to the cache directory. This path is used to store the task queue.
        If not provided, the 'CHESPEX_CACHE' environment variable is used, or the default
        '~/.chespex/cache' directory is used.
    """
    if task_name is None:
        return
    if not isinstance(task_name, list):
        task_name = [task_name]
    if cache_path is not None:
        cache_path = Path(cache_path)
    elif "CHESPEX_CACHE" in os.environ:
        cache_path = Path(os.environ["CHESPEX_CACHE"])
    else:
        cache_path = Path("~/.chespex/cache").expanduser()
    queue_filename = cache_path / "task_queue.txt"
    lock_filename = cache_path / "task_queue.lock"
    _wait_for_lock_file(lock_filename)
    with open(queue_filename, "a", encoding="utf-8") as queue_list_file:
        for task in task_name:
            try:
                task = Path(task).resolve()
                # print(f"Adding task {task} to queue")
                queue_list_file.write(str(task) + "\n")
            except ValueError:
                print(f"ValueError: {task} is not a valid path")
    _remove_lock_file(lock_filename)


def get_simulation_task_from_queue(cache_path=None):
    """
    Get the next simulation task from the queue.
    :param cache_path: Path to the cache directory. This path is used to store the task queue.
        If not provided, the 'CHESPEX_CACHE' environment variable is used, or the default
        '~/.chespex/cache' directory is used.
    :return: Filename of the next simulation task to run, or None if the queue is empty
        or does not exist.
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
    elif "CHESPEX_CACHE" in os.environ:
        cache_path = Path(os.environ["CHESPEX_CACHE"])
    else:
        cache_path = Path("~/.chespex/cache").expanduser()
    queue_filename = cache_path / "task_queue.txt"
    if not os.path.exists(queue_filename):
        return None
    lock_filename = cache_path / "task_queue.lock"
    _wait_for_lock_file(lock_filename)
    with open(queue_filename, "r+", encoding="utf-8") as queue_list_file:
        task_list = queue_list_file.readlines()
        if len(task_list) == 0:
            _remove_lock_file(lock_filename)
            return None
        task_name = task_list[0].strip()
        queue_list_file.seek(0)
        queue_list_file.truncate()
        for task in task_list[1:]:
            queue_list_file.write(task)
    _remove_lock_file(lock_filename)
    return task_name
