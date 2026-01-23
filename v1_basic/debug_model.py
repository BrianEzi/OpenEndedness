import jax
import jax.numpy as jnp
from models import ActorCritic
from env import BlindFetchEnv

def test_model():
    print("Testing ActorCritic init with Env...")
    env = BlindFetchEnv()
    rng = jax.random.PRNGKey(0)
    rng_env = jax.random.PRNGKey(1)
    
    obs = env.get_obs(env.reset(rng_env))
    print(f"Obs shapes: {jax.tree_util.tree_map(jnp.shape, obs)}")
    
    print("ActorCritic type:", type(ActorCritic))
    model = ActorCritic()
    print("Model type:", type(model))
    
    print("Calling init...")
    try:
        params = model.init(rng, obs)
        print("Init success!")
    except Exception as e:
        print(f"Init failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model()
