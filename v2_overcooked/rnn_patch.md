# Patch for Cook network (rnn.py)

The Cook agent lives inside JaxMARL, an external framework we do not push to this repo.
Two lines need to be changed in:

```
experiments/experiments/overcooked_v2_experiments/ppo/models/rnn.py
```

### Change 1 — line 58
Unpack `z_q` from the input tuple so the Cook receives Seer's message:

```python
# Before
obs, dones = x

# After
obs, dones, z_q = x
```

### Change 2 — after line 86 (after the LayerNorm)
Concatenate the Seer message into the Cook's CNN embedding before the GRU:

```python
embedding = nn.LayerNorm()(embedding)
embedding = jnp.concatenate([embedding, z_q], axis=-1)  # inject Seer message
```

This grows the embedding from (128,) to (136,) — the GRU then uses this as input.
The training loop (ippo.py) also needs to pass `z_q` through; Jonathan handles that.
