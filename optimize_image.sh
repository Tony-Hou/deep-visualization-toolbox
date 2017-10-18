#! /bin/bash

./optimize_image.py \
  --push-layer "inception_4c/pool_proj" \
  --output-template "inception_4c_%(p.push_channel)04d_%(p.rand_seed)d" \
  --push-channel 40 \
  --decay 0.0001 \
  --blur-radius 1.0 \
  --blur-every 4 \
  --small-norm-percentile 0 \
  --px-abs-benefit-percentile 0 \
  --max-iter 500 \
  --lr-policy "constant" \
  --lr-params "{'lr': 100.0}" \
  --data-size "224,224" \
  --rand-seed $RANDOM \
  --brave
