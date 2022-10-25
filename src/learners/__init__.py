from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .actor_critic_learner import ActorCriticLearner
from .maddpg_learner import MADDPGLearner
from .ppo_learner import PPOLearner
from .actor_critic_single_learner import ActorCriticSingleLearner
from .actor_critic_decentralized_learner import ActorCriticDecentralizedLearner
from .actor_critic_networked_learner import ActorCriticNetworkedLearner
from .actor_critic_networked_learner2 import ActorCriticNetworkedLearner2
from .actor_critic_networked_learner3 import ActorCriticNetworkedLearner3
from .actor_critic_regressor_learner import ActorCriticRegressorLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["actor_critic_learner"] = ActorCriticLearner
REGISTRY["maddpg_learner"] = MADDPGLearner
REGISTRY["ppo_learner"] = PPOLearner
REGISTRY["actor_critic_single_learner"] = ActorCriticSingleLearner
REGISTRY["actor_critic_decentralized_learner"] = ActorCriticDecentralizedLearner
REGISTRY["actor_critic_networked_learner"] = ActorCriticNetworkedLearner
REGISTRY["actor_critic_networked_learner2"] = ActorCriticNetworkedLearner2
REGISTRY["actor_critic_networked_learner3"] = ActorCriticNetworkedLearner3
REGISTRY["actor_critic_regressor_learner"] = ActorCriticRegressorLearner
