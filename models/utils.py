from enum import Enum


class ModelType(str, Enum):
    VideoTokenizer: str = 'VideoTokenizer'
    LatentActionModel: str = 'LatentAction'
    DynamicsModel: str = 'Dynamic'
    FlowDynamicsModel: str = 'FlowDynamic'
