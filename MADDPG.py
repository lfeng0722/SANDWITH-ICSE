import os
import torch as T
import torch.nn.functional as F
from agent import Agent

class MADDPG:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions,
                 scenario='simple', alpha=0.01, beta=0.02, fc1=128,
                 fc2=128, gamma=0.99, tau=0.01, chkpt_dir='tmp/maddpg/'):
        self.n_agents = n_agents
        self.n_actions = n_actions  # 6
        chkpt_dir += scenario

        self.agents = []
        for agent_idx in range(self.n_agents):
            self.agents.append(
                Agent(actor_dims[agent_idx], critic_dims,
                      n_actions, n_agents, agent_idx,
                      alpha=alpha, beta=beta,
                      fc1=fc1, fc2=fc2,
                      gamma=gamma, tau=tau,
                      chkpt_dir=chkpt_dir)
            )

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent in self.agents:
            os.makedirs(os.path.dirname(agent.actor.chkpt_file), exist_ok=True)
            agent.save_models()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()

    def choose_action(self, raw_obs):
        """
        对于每个Agent，调用agent.choose_action(...)。
        这里的返回 actions 将是 [n_agents, 6] 的 one-hot(6)，
        如果环境需要真正的动作离散index，可以对 each one-hot 做 argmax。
        """
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            # 返回 one-hot(6)
            action = agent.choose_action(raw_obs[agent_idx])
            actions.append(action)
        return actions  # shape: [n_agents, 6]

    def learn(self, memory, total_steps):
        if not memory.ready():
            return

        actor_states, states, actions, rewards, \
            actor_new_states, states_, dones = memory.sample_buffer()
        # memory 中 actions、new_actions 的 shape 应该是 [n_agents, batch_size, 6]

        device = self.agents[0].actor.device

        states  = T.tensor(states,  dtype=T.float).to(device)   # [batch_size, critic_dims]
        actions = T.tensor(actions, dtype=T.float).to(device)   # [n_agents, batch_size, 6]
        rewards = T.tensor(rewards, dtype=T.float).to(device)   # [batch_size, n_agents]
        states_ = T.tensor(states_, dtype=T.float).to(device)   # [batch_size, critic_dims]
        dones   = T.tensor(dones).to(device)                    # [batch_size, 1]

        # 准备目标Actor的“新动作”、以及旧动作
        all_agents_new_actions = []
        old_agents_actions = []

        for agent_idx, agent in enumerate(self.agents):
            new_states = T.tensor(actor_new_states[agent_idx], dtype=T.float).to(device)
            # target actor输出 logits
            new_pi_logits = agent.target_actor.forward(new_states)  # [batch_size, 6]
            # 用Gumbel-Softmax得到 one-hot(6)
            new_pi_one_hot = F.gumbel_softmax(new_pi_logits, tau=1.0, hard=True)
            all_agents_new_actions.append(new_pi_one_hot)

            # 旧动作也就是回放池里的 action
            old_agents_actions.append(actions[agent_idx])  # [batch_size, 6]

        # 拼接：先把list的每个 [batch_size,6] 沿dim=1拼 => [batch_size, n_agents*6]
        new_actions = T.cat(all_agents_new_actions, dim=1)  # [batch_size, n_agents*6]
        old_actions = T.cat(old_agents_actions, dim=1)      # [batch_size, n_agents*6]

        # 逐个agent更新
        for agent_idx, agent in enumerate(self.agents):
            # ---------- 1. 训练 Critic ----------
            with T.no_grad():
                critic_value_ = agent.target_critic.forward(states_, new_actions).flatten()
                # target = r + gamma * Q'(s', a')    (如果 done 则不加那一项)
                target = rewards[:, agent_idx] + (1 - dones[:, 0].float()) * agent.gamma * critic_value_

            critic_value = agent.critic.forward(states, old_actions).flatten()
            critic_loss = F.mse_loss(target, critic_value)

            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()
            agent.critic.scheduler.step()

            # ---------- 2. 训练 Actor ----------
            # 先复制一份 old_actions，然后用当前Actor的输出替换该agent那一段
            mu_states = T.tensor(actor_states[agent_idx], dtype=T.float).to(device)
            mu_logits = agent.actor.forward(mu_states)  # [batch_size, 6]

            # 用 Gumbel-Softmax(硬) 近似离散动作 => one-hot
            mu_one_hot = F.gumbel_softmax(mu_logits, tau=1.0, hard=True)

            oa = old_actions.clone()  # shape [batch_size, n_agents*6]
            start = agent_idx * self.n_actions
            end   = start + self.n_actions
            oa[:, start:end] = mu_one_hot

            # Actor希望最大化 Q(s, \tilde{a}), 这里取负号做梯度下降
            actor_loss = -T.mean(agent.critic.forward(states, oa).flatten())

            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor.optimizer.step()
            agent.actor.scheduler.step()

        # ---------- 软更新所有Agent的target网络 ----------
        for agent in self.agents:
            agent.update_network_parameters()
