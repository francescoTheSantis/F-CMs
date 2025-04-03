from os import environ as env
from pathlib import Path

PROJECT_NAME = "causal_rag"
WANDB_ENTITY = "causal_usi_supsi" # specify your wandb identity
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
 
HUGGINGFACEHUB_TOKEN='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'
OPENAI_API_KEY='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'
env['HYDRA_FULL_ERROR'] = '1'