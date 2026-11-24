# Task: Fix KB asset defects found by the eval case audit

## Context
The eval case audit (`evals/README.md`, Case audit) checks every case against its asset. Where a
case is sound and the asset is at fault, the case is left alone and the asset defect is listed
here, so the fix can come with an eval case that fails before it.

## Problem
