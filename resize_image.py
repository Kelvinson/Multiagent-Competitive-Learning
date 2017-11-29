from os import listdir
from os.path import isfile, join
import sys
import os

currentdir = os.path.join(os.getcwd(), '1')
print(currentdir)
files  = [f for f in listdir(currentdir)]
print(files)