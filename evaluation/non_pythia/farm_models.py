import threading
import queue
import torch
import os
import time
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import pickle

import token_loss
import sys


def main():
    if len(sys.argv) != 3:
        print("Usage: python farm_models.py model_list part(1 or 2)")
        return
    model_file = sys.argv[1]
    part = int(sys.argv[2])
    assert part in [1, 2, 5]

    HF_KEY = os.environ.get("HF_KEY")
    assert len(HF_KEY), "Set HF_KEY"

    # Configuration
    out_path = "output"
    model_list = pd.read_csv('model_list.csv')

    n = len(model_list)
    cut = int(n /2)

    if part in [1, 5]:
        print("Doing first half of model list.")
        model_list = model_list[:cut]
    else:
        print("Doing second half of model list.")
        model_list = model_list[cut:]

    threads_per_gpu = 2 # Adjust based on your GPU memory and workload
    max_gpu = 10  # debug to limit
    n_gpus = torch.cuda.device_count()

    os.makedirs(out_path, exist_ok=True)
    # Make filenames
    filenames = []
    for record in model_list.to_dict('records'):
        display = record['name'].strip().replace("/", "").replace(" ", "_")
        filenames.append(f"{out_path}/{display}.pkl")
    model_list["filename"] = filenames 
    model_list.drop(columns=['num_parameters'], inplace=True)
    model_list['url'] = [v.strip() for v in model_list['url']]

    # preload the data
    data = token_loss.load_data()
    data["HF_KEY"] = HF_KEY

    # Detect available GPUs
    if n_gpus == 0:
        print("No GPUs detected, falling back to CPU")
        n_gpus = 1  # CPU-only mode
    n_gpus = min(n_gpus, max_gpu)

    if part==5:
        # test single threaded first
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        row = model_list.to_dict('records')[0]
        token_loss.process(**row, **data, device=device)
        return


    total_threads = threads_per_gpu * n_gpus

    # Create job and result queues
    job_queue = queue.Queue()

    for job_dict in model_list.to_dict('records'):
        job_queue.put(job_dict)

    # Add terminate signal (note: we could also add a sync signal)
    for _ in range(total_threads):
        job_queue.put(None)

    # Wait for all jobs to be completed
    print(f"Waiting for {len(model_list)} jobs to complete...")


    print("Starting workers")
    # Create and start worker threads
    workers = []
    for i in range(total_threads):
        # Assign GPU ID based on round-robin
        gpu_id = i % n_gpus

        # Create worker
        worker = GPUWorker(
            data=data,
            worker_id=i,
            gpu_id=gpu_id,
            job_queue=job_queue,
        )
        worker.start()
        workers.append(worker)
        print(f"Started worker {i} assigned to GPU {gpu_id}")

    job_queue.join()
    print("All jobs completed!")



class GPUWorker(threading.Thread):
    """Worker thread assigned to a specific GPU"""
    def __init__(self, 
                 worker_id: int, 
                 gpu_id: int,
                 job_queue: queue.Queue,
                 data):
        super().__init__()
        self.worker_id = worker_id
        self.gpu_id = gpu_id
        self.job_queue = job_queue
        self.data = data
        self.daemon = True  # Daemon threads exit when the main thread exits


    def run(self):
        """Main worker loop"""

        # switch to device
        if torch.cuda.is_available():
            device = f"cuda:{self.gpu_id}"
        else:
            device = "cpu"

        consts = dict(
            device=torch.device(device),
            **self.data
        )

        print(f"Worker {self.worker_id} started on {device}")
        time.sleep(.5)  # give things time to settle down
        while True:
            try:
                # Get a job from the queue
                job = self.job_queue.get(block=True, timeout=5)

                if job is None:  # Sentinel value to signal thread termination
                    self.job_queue.task_done()
                    break

                # Process the job
                try:
                    token_loss.process(**job, **consts)
                except Exception as e:
                    print(f"Worker {self.worker_id} (GPU {self.gpu_id}) failed job for model {job['name']}: {str(e)}")
                finally:
                    # Mark the job as done
                    self.job_queue.task_done()

            except Exception as e:
                print(f"Worker {self.worker_id} encountered an unexpected error: {str(e)}")
                break


if __name__ == "__main__":
    main()
