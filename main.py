# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from jax.lib import xla_bridge
import jax
from jax import config; config.update("jax_enable_x64", True)

print(jax.devices())
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)

import jax.numpy as jnp
