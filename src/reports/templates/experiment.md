---
id: $id
title: $title
kind: experiment
date: $date
status: planned            # planned | running | complete | abandoned | superseded
branch: $branch
pr:                        # filled in when the pull request exists
commits: []
related: []                # ids of related records, e.g. [EXP-0003, CHG-0011]
supersedes:                # id of the record this replaces, if any
tags: []
publication:
  target:                  # which paper or thesis chapter this feeds
  section:
  figures: []
hypothesis: >
  One sentence: what this experiment would show if it worked.
data:
  - name:                  # dataset or run this consumed
    path:
    n:                     # number of samples
    patients:
    split:                 # train | dev | test | pool
    leakage:               # one line: why the held-out data stayed held out
compute:
  device:
  wall_time_h:
  mc_seconds: 0            # Monte Carlo budget spent, the currency of this project
commands:
  - # the exact command that reproduces this
metrics:
  - {name: , value: , unit: , split: , n: }
artifacts:
  run_dir:
  figures: []
  tables: []
---

## What was done

## Methodology

## Data used

## What precisely changed

## Results: positive

## Results: negative

## What went well

## What went wrong

## Next steps
