from contextlib import contextmanager
import os
import time
from agentenv.envs import (
    AcademiaEnvClient,
    AlfWorldEnvClient,
    BabyAIEnvClient,
    MazeEnvClient,
    MovieEnvClient,
    SciworldEnvClient,
    SheetEnvClient,
    SqlGymEnvClient,
    TextCraftEnvClient,
    TodoEnvClient,
    WeatherEnvClient,
    WebarenaEnvClient,
    WebshopEnvClient,
    WordleEnvClient,
    SearchQAEnvClient,
)

import torch.distributed as dist

def _select_env_addr(env_addr_value):
    """
    Selects an environment address from a comma-separated list based on the distributed rank.
    """
    addrs = [a.strip() for a in str(env_addr_value).split(",") if a.strip()]
    if len(addrs) <= 1:
        return addrs[0] if addrs else env_addr_value
    
    # Select address based on rank to distribute load across multiple servers
    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = int(os.environ.get("RANK", 0))
    
    selected_addr = addrs[rank % len(addrs)]
    print(f"[Rank {rank}] Selected env_addr: {selected_addr}")
    return selected_addr

def init_env_client(args):
    # task_name - task dict
    envclient_classes = {
        "webshop": WebshopEnvClient,
        "alfworld": AlfWorldEnvClient,
        "babyai": BabyAIEnvClient,
        "sciworld": SciworldEnvClient,
        "textcraft": TextCraftEnvClient,
        "webarena": WebarenaEnvClient,
        "sqlgym": SqlGymEnvClient,
        "maze": MazeEnvClient,
        "wordle": WordleEnvClient,
        "weather": WeatherEnvClient,
        "todo": TodoEnvClient,
        "movie": MovieEnvClient,
        "sheet": SheetEnvClient,
        "academia": AcademiaEnvClient,
        "searchqa": SearchQAEnvClient,
    }
    # select task according to the name
    envclient_class = envclient_classes.get(args.task_name.lower(), None)
    if envclient_class is None:
        raise ValueError(f"Unsupported task name: {args.task_name}")
    
    # Handle multiple comma-separated environment addresses
    selected_env_addr = _select_env_addr(args.env_addr)
    
    retry = 0
    while True:
        try:
            env_client = envclient_class(env_server_base=selected_env_addr, data_len=1, timeout=2400)
            break
        except Exception as e:
            retry += 1
            print(f"Failed to connect to env server {selected_env_addr}, retrying...({retry}/{args.max_retries})")
            if retry > args.max_retries:
                raise e
            time.sleep(5)
    return env_client