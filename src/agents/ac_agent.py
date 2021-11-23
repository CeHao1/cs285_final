from collections import OrderedDict

from src.critics.bootstrapped_continuous_critic import \
    BootstrappedContinuousCritic
from src.infrastructure.replay_buffer import ReplayBuffer
from src.infrastructure.utils import *
from src.policies.MLP_policy import MLPPolicyAC
from src.infrastructure import pytorch_util as ptu
from .base_agent import BaseAgent

from src.infrastructure.nn_manager import save_nn_frame, load_nn_frame

class ACAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(ACAgent, self).__init__()

        self.env = env
        self.agent_params = agent_params

        self.gamma = self.agent_params['gamma']
        self.standardize_advantages = self.agent_params['standardize_advantages']

        self.actor = MLPPolicyAC(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['discrete'],
            self.agent_params['learning_rate'],
        )
        self.critic = BootstrappedContinuousCritic(self.agent_params)

        self.replay_buffer = ReplayBuffer()

    def save_agent(self, name):
        save_nn_frame(self.actor, nn_type='actor', name=name)
        save_nn_frame(self.critic, nn_type='critic', name=name)

    def load_agent(self, name):
        self.actor = load_nn_frame(nn_type='actor', name=name)
        self.critic = load_nn_frame(nn_type='critic', name=name)

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        # for agent_params['num_critic_updates_per_agent_update'] steps,
        #     update the critic
        loss = OrderedDict()

        for i in range(self.agent_params['num_critic_updates_per_agent_update']):
            loss['Critic_Loss'] = self.critic.update(ob_no, ac_na, next_ob_no, re_n, terminal_n)

        advantages = self.estimate_advantage(ob_no, next_ob_no, re_n, terminal_n)

        # for agent_params['num_actor_updates_per_agent_update'] steps,
        #     update the actor
        for i in range(self.agent_params['num_actor_updates_per_agent_update']):
            loss['Actor_Loss'] = self.actor.update(ob_no, ac_na, advantages)
            
        return loss

    def estimate_advantage(self, ob_no, next_ob_no, re_n, terminal_n):
        # 1) query the critic with ob_no, to get V(s)
        # 2) query the critic with next_ob_no, to get V(s')
        # 3) estimate the Q value as Q(s, a) = r(s, a) + gamma*V(s')
        # HINT: Remember to cut off the V(s') term (ie set it to 0) at terminal states (ie terminal_n=1)
        # 4) calculate advantage (adv_n) as A(s, a) = Q(s, a) - V(s)

        ob_no = ptu.from_numpy(ob_no)
        next_ob_no = ptu.from_numpy(next_ob_no)
        re_n = ptu.from_numpy(re_n)
        terminal_n = ptu.from_numpy(terminal_n)

        v_s = self.critic(ob_no)
        v_sp = self.critic(next_ob_no)
        q = re_n + self.gamma * v_sp * (1 - terminal_n)
        adv_n = ptu.to_numpy(q - v_s)

        if self.standardize_advantages:
            adv_n = (adv_n - np.mean(adv_n)) / (np.std(adv_n) + 1e-8)
        return adv_n

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size)
