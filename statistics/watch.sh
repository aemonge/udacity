#!/bin/bash

ls *.py | entr -rn python3 "./bayes-rule.py"
