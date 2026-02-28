# `envs.origami_env`

Reinforcement-learning style environment for 2D origami folding.

## Classes

### `OrigamiEnv(resolution, max_steps)`

Environment that tracks current folded paper against a target mask.

### Parameters

| Name | Type | Description |
| --- | --- | --- |
| `resolution` | `tuple[int, int]` | Observation and target mask shape `(H, W)`. |
| `max_steps` | `int` | Episode truncation limit. |

### Methods

>#### `reset(context) -> np.ndarray`
>
>Initializes environment state.
>
>**Parameters**
>
>| Name | Type | Description |
>| --- | --- | --- |
>| `context` | `dict` | Context payload with `base_paper` and `target_mask`. |
>
>Context keys:
>
>| Key | Type | Description |
>| --- | --- | --- |
>| `base_paper` | `paper.Paper` | Initial folded state. |
>| `target_mask` | `np.ndarray` | Boolean mask with shape `(H, W)`. |
>
>**Returns**
>
>| Type | Description |
>| --- | --- |
>| `np.ndarray` | Observation tensor with shape `(2, H, W)`, dtype `float32`. Channel 0 is current paper mask; channel 1 is target mask. |

>#### `step(action) -> tuple[np.ndarray, float, bool, bool, dict]`
>
>Applies one fold action.
>
>**Parameters**
>
>| Name | Type | Description |
>| --- | --- | --- |
>| `action` | `np.ndarray` | Fold endpoints `[x1, y1, x2, y2]`. |
>
>**Returns**
>
>Tuple values in order `(obs, reward, done, truncated, info)`:
>
>| Position | Name | Description |
>| --- | --- | --- |
>| 0 | `obs` | `np.ndarray` with shape `(2, H, W)` and dtype `float32`. |
>| 1 | `reward` | Fold validity and terminal IoU reward. |
>| 2 | `done` | `True` when IoU exceeds threshold. |
>| 3 | `truncated` | `True` when `max_steps` is reached. |
>| 4 | `info` | Extra metadata dictionary (currently empty). |

### Examples

```python
import numpy as np
from paper import Paper
from envs.origami_env import OrigamiEnv

env = OrigamiEnv((64, 64), max_steps=5)
obs = env.reset({"base_paper": Paper(), "target_mask": np.ones((64, 64), dtype=bool)})
obs, reward, done, truncated, info = env.step(np.array([0.0, 0.0, 1.0, 0.5], dtype=np.float32))
print(obs.shape, type(reward), done, truncated, type(info).__name__)
```

Expected output (example):

```text
(2, 64, 64) <class 'float'> False False dict
```
