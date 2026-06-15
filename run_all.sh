#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
python run_observed_pareto.py
python run_surrogate_nsga2.py
