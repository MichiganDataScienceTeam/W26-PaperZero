# `framework`

Base Python interfaces for samplers and environments.

## Type Aliases

| Alias | Type | Description |
| --- | --- | --- |
| `Context` | `dict[str, Any]` | Generic payload used to initialize an environment. |
| `StepResult` | `tuple[Any, float, bool, bool, dict[str, Any]]` | Step output `(observation, reward, done, truncated, info)`. |

## Classes

### `ContextSampler`

Interface for context generation.

### Methods

>#### `sample(level: int) -> Context`
>
>Samples one context payload for a requested difficulty level.
>
>**Parameters**
>
>| Name | Type | Description |
>| --- | --- | --- |
>| `level` | `int` | Difficulty or depth requested by the caller. |

>#### `update(metrics: dict[str, Any]) -> None`
>
>Optional feedback hook for adaptive samplers.
>
>**Parameters**
>
>| Name | Type | Description |
>| --- | --- | --- |
>| `metrics` | `dict[str, Any]` | Metrics from training or evaluation. |

### `Environment`

Interface for interactive environments.

### Methods

>#### `reset(context: Context) -> Any`
>
>Resets environment state from a context payload.

>#### `step(action: Any) -> StepResult`
>
>Applies one action and returns `(obs, reward, done, truncated, info)`.

### Examples

```python
from framework import ContextSampler, Environment

class MySampler(ContextSampler):
    def sample(self, level: int):
        return {"level": level}

class MyEnv(Environment):
    def reset(self, context):
        return context
    def step(self, action):
        return action, 0.0, False, False, {}

sampler = MySampler()
env = MyEnv()
print(sampler.sample(2))
print(env.step({"a": 1}))
```

Expected output:

```text
{'level': 2}
({'a': 1}, 0.0, False, False, {})
```
