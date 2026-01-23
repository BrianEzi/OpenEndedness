import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
import distrax
import chex
from typing import NamedTuple, Any

from env import BlindFetchEnv, EnvState
from models import ActorCritic

class Transition(NamedTuple):
    obs: dict
    action: dict
    reward: jnp.ndarray
    done: jnp.ndarray
    value: jnp.ndarray
    log_prob: jnp.ndarray
    success: jnp.ndarray
    msg_norm: jnp.ndarray
    doer_entropy: jnp.ndarray

def make_train(config):
    env = BlindFetchEnv()
    
    def train(rng):
        # Init env
        rng, rng_env = jax.random.split(rng)
        init_obs = env.get_obs(env.reset(rng_env)) # Single env init for shape
        
        # Init network
        network = ActorCritic()
        rng, rng_net = jax.random.split(rng)
        init_params = network.init(rng_net, init_obs)
        

        # Let's use the standard chain with scale_by_adam and scale_by_schedule?
        # Simpler: optax.adam(learning_rate=schedule)
        schedule = optax.linear_schedule(
            init_value=3e-4, 
            end_value=0.0, 
            transition_steps=config["num_updates"]
        )
        tx = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adam(learning_rate=schedule)
        )

        train_state = TrainState.create(
            apply_fn=network.apply,
            params=init_params,
            tx=tx
        )

        # vectorized env
        batch_reset = jax.vmap(env.reset)
        batch_step = jax.vmap(env.step)
        batch_get_obs = jax.vmap(env.get_obs)

        rng, rng_envs = jax.random.split(rng)
        rng_envs = jax.random.split(rng_envs, config["num_envs"])
        env_state = batch_reset(rng_envs)
        
        # Initial observation
        curr_obs = batch_get_obs(env_state)
        
        # Train Loop
        
        def _update_step(runner_state, unused):
            train_state, env_state, curr_obs, rng = runner_state
            
            # ROLLOUT
            def _env_step(carry, unused):
                env_state, last_obs, rng = carry
                rng, rng_act, rng_step = jax.random.split(rng, 3)
                
                # Action selection
                seer_dist, doer_dist, value = train_state.apply_fn(train_state.params, last_obs)
                
                seer_msg = seer_dist.sample(seed=rng_act)
                doer_move = doer_dist.sample(seed=rng_act)
                
                seer_log_prob = seer_dist.log_prob(seer_msg)
                doer_log_prob = doer_dist.log_prob(doer_move)
                
                # Sum log probs? They are independent actions.
                # Total log prob = log(P(seer)) + log(P(doer))
                log_prob = seer_log_prob + doer_log_prob
                
                actions = {'seer_msg': seer_msg, 'doer_move': doer_move}
                
                # Env step
                rng_step = jax.random.split(rng_step, config["num_envs"])
                next_env_state, reward, done, info = batch_step(env_state, actions, rng_step)
                
                # Auto-reset
                # If done, next_env_state should be reset state.
                # But typically we handle this by masking.
                # However, for pure JAX auto-reset is cleaner.
                # Let's implement auto-reset where done:
                # But we need fresh keys for reset.
                rng_reset = jax.random.split(rng_step[0], config["num_envs"]) # Simple split hack?
                # A better way: Pass rng into step for usage? 
                # Our env reset needs rng.
                
                # Let's use standard pattern:
                # if done, reset.
                reset_state = batch_reset(rng_reset)
                
                # Select based on done
                # done is (num_envs,) bool
                # env_state fields: (num_envs, ...)
                
                def where_state(c, s1, s2):
                    return jax.tree_util.tree_map(lambda x, y: jnp.where(c.reshape(-1, *([1]*(x.ndim-1))), x, y), s1, s2)
                
                next_env_state = where_state(done, reset_state, next_env_state)
                
                next_obs = batch_get_obs(next_env_state)
                
                # Metrics
                msg_norm = jnp.linalg.norm(seer_msg, axis=-1)
                doer_entropy = doer_dist.entropy()
                
                transition = Transition(
                    obs=last_obs,
                    action=actions,
                    reward=reward,
                    done=done,
                    value=value,
                    log_prob=log_prob,
                    success=info['reached_target'],
                    msg_norm=msg_norm,
                    doer_entropy=doer_entropy
                )
                
                return (next_env_state, next_obs, rng), transition

            carry = (env_state, curr_obs, rng)
            (env_state, curr_obs, rng), traj_batch = jax.lax.scan(
                _env_step, carry, None, length=config["num_steps"]
            )
            
            # CALCULATE GAE
            # Need value of next_obs (bootstrap)
            _, _, next_val = train_state.apply_fn(train_state.params, curr_obs)
            
            def calculate_gae(traj_batch, next_val):
                # (steps, envs)
                rewards = traj_batch.reward
                dones = traj_batch.done
                values = traj_batch.value
                
                # Combine last val
                # append next_val to values to handle t+1
                # rewards: [0...T-1]
                # values: [0...T-1]
                # next_val: [T]
                
                advantages = []
                gae = 0.0
                for t in reversed(range(config["num_steps"])):
                    # V(t+1)
                    val_t1 = next_val if t == config["num_steps"] - 1 else values[t+1]
                    # mask
                    mask = 1.0 - dones[t]
                    
                    delta = rewards[t] + config["gamma"] * val_t1 * mask - values[t]
                    gae = delta + config["gamma"] * config["gae_lambda"] * mask * gae
                    advantages.insert(0, gae)
                    
                advantages = jnp.array(advantages)
                returns = advantages + values
                return advantages, returns

            advantages, targets = calculate_gae(traj_batch, next_val)
            
            # LOSS & UPDATE
            def loss_fn(params, traj_batch, advantages, targets):
                obs = traj_batch.obs
                actions = traj_batch.action
                
                seer_dist, doer_dist, value = train_state.apply_fn(params, obs)
                
                # Log Probs
                seer_log = seer_dist.log_prob(actions['seer_msg'])
                doer_log = doer_dist.log_prob(actions['doer_move'])
                new_log_prob = seer_log + doer_log
                
                # Ratio
                log_ratio = new_log_prob - traj_batch.log_prob
                ratio = jnp.exp(log_ratio)
                
                # Clip Loss
                adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                pg_loss1 = -adv * ratio
                pg_loss2 = -adv * jnp.clip(ratio, 1.0 - 0.2, 1.0 + 0.2)
                pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()
                
                # Value Loss
                v_loss = jnp.square(value - targets).mean()
                
                # Entropy
                entropy = seer_dist.entropy().mean() + doer_dist.entropy().mean()
                
                loss = pg_loss + 0.5 * v_loss - 0.01 * entropy
                return loss, (pg_loss, v_loss, entropy)

            # Flatten batch for update (steps * envs, ...)
            # Handled automatically by JAX vmap usually if we want minibatching.
            # But here prompt says: "Update: optax.chain..."
            # Usually we iterate over minibatches.
            # "Steps per Update: 128" -> This usually means rollout length?
            # "Num Envs: 256"
            # Batch size = 128 * 256 = 32768.
            # Single update step on full batch? Or minibatches?
            # "Steps per Update" typically implies rollout length.
            # Given the simple prompt, one update per rollout is likely intended/sufficient.
            # I will just do one update on the whole batch for simplicity and speed (it fits in GPU memory usually).
            # (128 * 256 * obs_size is small).

            loss_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss_val, (pg_loss, v_loss, ent)), grads = loss_grad_fn(
                train_state.params, traj_batch, advantages, targets
            )
            
            train_state = train_state.apply_gradients(grads=grads)
            
            metrics = {
                "loss": loss_val,
                "pg_loss": pg_loss,
                "v_loss": v_loss,
                "entropy": ent,
                "reward_mean": traj_batch.reward.mean(),
                "success_rate": traj_batch.success.mean(), # Approximation, better if normalized by dones if possible
                "msg_magnitude": traj_batch.msg_norm.mean(),
                "doer_entropy": traj_batch.doer_entropy.mean()
            }
            
            return (train_state, env_state, curr_obs, rng), metrics

        runner_state = (train_state, env_state, curr_obs, rng)
        
        num_updates = config["num_updates"]
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, length=num_updates
        )
        
        final_train_state = runner_state[0]
        return final_train_state, metrics

    return train
