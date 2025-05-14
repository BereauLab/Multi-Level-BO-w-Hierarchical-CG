"""Simulator file for running GROMACS simulations from the simulation queue."""

from typing import Optional
import os
from time import sleep
import subprocess
import argparse
from datetime import datetime
import socket
from .helper import get_simulation_task_from_queue, add_simulation_task_to_queue


def _reset_task(task_name: str):
    """
    Delete task output files if it wasn't finished successfully. Afterwards, add it back to
    the queue.
    :param task_name: The name of the task to reset.
    """
    if task_name is None:
        return
    parent_directory = os.path.dirname(task_name)
    base_name = os.path.basename(task_name)
    if os.path.exists(parent_directory) and not os.path.exists(f"{task_name}.gro"):
        for file in os.listdir(parent_directory):
            if (
                ".tpr" not in file
                and ".mdp" not in file
                and "log" not in file
                and base_name in file
            ):
                os.remove(os.path.join(parent_directory, file))
    add_simulation_task_to_queue(task_name)


def run_simulations_from_queue(extra: str = "", max_tasks: Optional[int] = None):
    """
    Run GROMACS simulations from the simulation queue.
    """
    # Parse command line arguments
    ap = argparse.ArgumentParser(
        description="Run GROMACS simulations from the simulation queue."
    )
    ap.add_argument(
        "--extra", "-e", type=str, default="", help="Extra arguments to pass to mdrun."
    )
    ap.add_argument(
        "--max-tasks",
        "-nt",
        type=int,
        default=None,
        help="Maximum number of tasks to run before exiting.",
    )
    ap.add_argument(
        "--auto-quit",
        "-q",
        action="store_true",
        help="Automatically quit if the queue is empty.",
    )
    args = ap.parse_args()
    extra = args.extra if args.extra is not None else ""
    max_tasks = args.max_tasks if args.max_tasks is not None else None
    auto_quit = args.auto_quit
    # Run simulations
    quit_counter = 0
    while max_tasks is None or max_tasks > 0:
        try:
            task = get_simulation_task_from_queue()
            if task is not None:
                quit_counter = 0
                timestamp = str(datetime.now()).split(".", maxsplit=1)[0]
                with open(f"{task}.run.log", "w", encoding="utf-8") as log_file:
                    hostname = socket.gethostname()
                    log_file.write(
                        f"Simulation started at {timestamp} on host {hostname}\n"
                    )
                basename = os.path.basename(task)
                dirname = os.path.dirname(task)
                print(f"{timestamp} Running task: {task}")
                command = (
                    f"gmx mdrun -deffnm {basename} {extra} >> {basename}.run.log 2>&1"
                )
                subprocess.run(command, shell=True, check=True, cwd=dirname)
                if not os.path.exists(f"{task}.gro"):
                    raise RuntimeError("Task failed: Gro file not found.")
                timestamp = str(datetime.now()).split(".", maxsplit=1)[0]
                print(f"{timestamp} Task finished: {task}")
                if max_tasks is not None:
                    max_tasks -= 1
            else:
                if auto_quit:
                    quit_counter += 1
                    if quit_counter > 5:
                        print("Queue is empty. Exiting...")
                        break
                sleep(2)
        except KeyboardInterrupt:
            print("Exiting...")
            _reset_task(task)
            break
        except Exception as e:  # pylint: disable=broad-except
            print(e)
            _reset_task(task)
            try:
                sleep(10)
            except KeyboardInterrupt:
                print("Exiting...")
                break
