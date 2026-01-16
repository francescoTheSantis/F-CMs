from os import environ as env
from pathlib import Path

PROJECT_NAME = "federated_c2bm"
WANDB_ENTITY = "" # specify your wandb identity
CACHE = Path(
    env.get(
        f"{PROJECT_NAME.upper()}_CACHE",
        Path(
            env.get("XDG_CACHE_HOME", Path("~", ".cache")),
            PROJECT_NAME,
        ),
    )
).expanduser()
CACHE.mkdir(exist_ok=True)

HUGGINGFACEHUB_TOKEN=''    # set your huggingface token here
OPENAI_API_KEY='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'    # set your openai api key here
 
env['HYDRA_FULL_ERROR'] = '1'
env['CUDA_LAUNCH_BLOCKING'] = '1'
env['TOKENIZERS_PARALLELISM'] = 'false'
