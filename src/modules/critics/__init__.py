from .centralV import CentralVCritic
from .centralV_ns import CentralVCriticNS
from .ac import ACCritic
from .ac_ns import ACCriticNS
from .ac_networked import ACCriticNetworked

REGISTRY = {}

REGISTRY["cv_critic"] = CentralVCritic
REGISTRY["cv_critic_ns"] = CentralVCriticNS
REGISTRY["ac_critic"] = ACCritic
REGISTRY["ac_critic_ns"] = ACCriticNS
REGISTRY["ac_critic_networked"] = ACCriticNetworked


