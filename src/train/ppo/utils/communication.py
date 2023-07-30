"""

Author: Sa1g
Github: https://github.com/sa1g

Date: 2023.07.30

"""
import os
import pickle


def save_batch(data, worker_id):
    """
    Save the batch in ram as a binary.
    It uses `pickle.HIGHEST_PROTOCOL`.

    Args:
        data: the batch
        path: worker_id.
    """
    with open(f"/dev/shm/{worker_id}.bin", "wb") as data_file:
        pickle.dump(data, data_file, pickle.HIGHEST_PROTOCOL)


def load_batch(worker_id):
    """
    Loads the batch from ram.
    It uses `pickle.HIGHEST_PROTOCOL`.

    Args:
        path: worker_id.

    Returns:
        batch
    """
    with open(f"/dev/shm/{worker_id}.bin", "rb") as data_file:
        return pickle.load(data_file)


def delete_batch(workers_id: list):
    """
    Deletes workers batch from ram.
    """
    for _id in workers_id:
        os.remove(f"/dev/shm/{_id}.bin")


def data_logging(data, experiment_id, id):
    """
    Saves data in /experiments/exp_ID/logs/{id}.csv

    Args:
        data: the data you want to save, preformatted
        experiment_id: experiment full name
        id: process_id
    """
    with open(f"experiments/{experiment_id}/logs/{id}.csv", "a+") as file:
        file.write(data)
