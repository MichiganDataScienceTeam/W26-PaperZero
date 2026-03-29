# `data.origami_sampler`

Sampling utilities for generating origami fold tasks.

## Classes

### `OrigamiSampler`

Stateless sampler that produces a folded paper trajectory summary.

### Parameters

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `max_fold_attempts` | `int` | `20` | Maximum random fold-line attempts per fold step. |
| `max_paper_retries` | `int` | `10` | Maximum retries at each step before moving on. |

### Methods

>#### `sample(level: int) -> dict[str, Any]`
>
>Generates a folded paper task with up to `level` successful folds.
>
>**Parameters**
>
>| Name | Type | Description |
>| --- | --- | --- |
>| `level` | `int` | Maximum number of fold steps to attempt. |
>
>**Returns**
>
>Returns a dictionary with the following keys:
>
>| Key | Type | Description |
>| --- | --- | --- |
>| `total_action` | `np.ndarray` | Flattened fold endpoints `[x1, y1, x2, y2, ...]`; shape `(4 * actual_folds,)`. |
>| `actual_folds` | `int` | Number of successful folds. |
>| `final_paper` | `paper.Paper` | Folded paper state after sampling. |

### Examples

```python
from data.origami_sampler import OrigamiSampler

sampler = OrigamiSampler()
sample = sampler.sample(level=3)
print(sample["actual_folds"], sample["total_action"].shape)
```

Expected output (example):

```text
3 (12,)
```

## Notes

- `actual_folds <= level`.
- If no fold succeeds, `total_action` is an empty array with shape `(0,)`.
