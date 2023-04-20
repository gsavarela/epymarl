from .coma import COMACritic
from .centralV import CentralVCritic
from .coma_ns import COMACriticNS
from .centralV_ns import CentralVCriticNS
from .maddpg import MADDPGCritic
from .maddpg_ns import MADDPGCriticNS
from .ac import ACCritic
from .ac_ns import ACCriticNS
from .ac_dec import ACCriticDecentralized
from .ac_networked import ACCriticNetworked

REGISTRY = {}

REGISTRY["coma_critic"] = COMACritic
REGISTRY["cv_critic"] = CentralVCritic
REGISTRY["coma_critic_ns"] = COMACriticNS
REGISTRY["cv_critic_ns"] = CentralVCriticNS
REGISTRY["maddpg_critic"] = MADDPGCritic
REGISTRY["maddpg_critic_ns"] = MADDPGCriticNS
REGISTRY["ac_critic"] = ACCritic
REGISTRY["ac_critic_ns"] = ACCriticNS
REGISTRY["ac_critic_dec"] = ACCriticDecentralized
REGISTRY["ac_critic_networked"] = ACCriticNetworked


