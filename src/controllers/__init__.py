REGISTRY = {}

from .basic_controller import BasicMAC
from .non_shared_controller import NonSharedMAC
from .maddpg_controller import MADDPGMAC
from .single_agent_controller import SAC
from .decentralized_agent_controller import DAC
from .pic_controller import PICMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["non_shared_mac"] = NonSharedMAC
REGISTRY["maddpg_mac"] = MADDPGMAC
REGISTRY["sac"] = SAC
REGISTRY["dac"] = DAC
REGISTRY["pic_mac"] = PICMAC
