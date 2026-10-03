import os
from models.utils import ModelType
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh, DeviceMesh
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import fully_shard
from typing import Dict, Iterable

from utils.config import DistributedConfig



def init_distributed_from_env() -> Dict[str, object]:
    """Initialize DeviceMesh from torchrun env vars.
    Returns a context dict with is_distributed, world_size, is_main, device_mesh.
    """
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    is_distributed = world_size > 1 and torch.cuda.is_available()

    if is_distributed:
        device_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=('fsdp',))
    else:
        device_mesh = None

    rank = device_mesh.get_rank() if is_distributed else 0
    is_main = (rank == 0)

    return {
        'is_distributed': is_distributed,
        'world_size': world_size,
        'is_main': is_main,
        'device_mesh': device_mesh,
    }


def prepare_model_for_distributed(model: torch.nn.Module, config: DistributedConfig, model_type: ModelType, device_mesh: DeviceMesh) -> torch.nn.Module:
    if config.use_ddp:
        return DDP(model, device_ids=[device_mesh.get_local_rank()], output_device=device_mesh.get_local_rank(), find_unused_parameters=False)
    if config.use_fsdp:
        mp_policy = config.get_mixed_precision_policy()
        fsdp_kwargs = {
            "reshard_after_forward": config.reshard_after_forward,
            "mp_policy": mp_policy,
            "offload_policy": config.offload_policy,
        }

        def shard_layers(layers: Iterable[torch.nn.Module]) -> None:
            for layer in layers:
                fully_shard(layer, mesh=device_mesh, **fsdp_kwargs)

        if model_type in {ModelType.VideoTokenizer, ModelType.LatentActionModel}:
            shard_layers(model.encoder.transformer.blocks)

            if model_type == ModelType.VideoTokenizer:
                shard_layers(model.encoder.latent_head)

            if model_type == ModelType.LatentActionModel:
                shard_layers(model.encoder.action_head)
                shard_layers(model.decoder.transformer.blocks)
                shard_layers(model.decoder.frame_head)

            fully_shard(model.encoder, mesh=device_mesh, **fsdp_kwargs)
            fully_shard(model.decoder, mesh=device_mesh, **fsdp_kwargs)
            fully_shard(model.quantizer, mesh=device_mesh, **fsdp_kwargs)

        elif model_type in {ModelType.DynamicsModel, ModelType.FlowDynamicsModel}:
            shard_layers(model.transformer.blocks)
            fully_shard(model.latent_embed, mesh=device_mesh, **fsdp_kwargs)
            fully_shard(model.output_mlp, mesh=device_mesh, **fsdp_kwargs)
        else:
            raise ValueError('Unknown model type')
        fully_shard(
            model,
            mesh=device_mesh,
            **fsdp_kwargs,
        )

    return model


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def print_param_count_if_main(model: torch.nn.Module, model_name: str, is_main: bool) -> None:
    if not is_main:
        return
    try:
        params = sum(p.numel() for p in model.parameters())
        print(f"{model_name} parameters: {params/1e6:.2f}M ({params})")
    except Exception:
        pass


def cleanup_distributed(is_distributed: bool) -> None:
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group() 
