#!/bin/bash
```
$1:job name
$2:node name
$3:fast or common
$4:model path, 'model3006450.h5' eg.
```

srun -J $1 -w $2  --gres=gpu:1 -p $3 --gres-flags=enforce-binding -s singularity exec -B /tmp:/run /atlas/home/zwpeng/containers/mitosis_detcetion-2017-0808.img python3 predict.py --model_path=$4

