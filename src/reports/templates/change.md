---
id: $id
title: $title
kind: change
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
compute:
  device:
  wall_time_h:
  mc_seconds: 0            # Monte Carlo budget spent, the currency of this project
commands:
  - # the exact command that reproduces this
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
