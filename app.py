#coding=utf-8
import feature
import sys, os, logging
import numpy as np


if len(sys.argv) > 1:
    name = sys.argv[1]
else :
    exit();
v = feature.feature(name)
print v.get_features()

