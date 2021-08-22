## Overview

| Package  | Usage  |  
|---|---| 
| pyecharts_tool | Build based on pyecharts, extends its functionality |  
| landborn | Extention of seaborn, developed based on matplotlib |
| starborn | Interactive data visualization tool. |

### Landborn

see `landborn.ipynb`.

    barplot_colorbar("name", "height", color="weight", data=df)
    barhplot_stacked("weight", "class", hue="gender", data=df)

### starborn

see `starborn.ipynb`.

```python
    iscatter("x", "y", data=df, signals=points)
    # iscatter("pre_rr", "post_rr", data=df, signals=heartbeats, custom_star=HeartbeatPoint)
```
