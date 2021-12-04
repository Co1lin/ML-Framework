import os
import sys

lines = os.popen(f'ps aux | grep {sys.argv[1]}').readlines()
print(lines)
for line in lines:
    line = line.split()
    p = line[1]
    print(f'killing {p}')
    os.system(f'kill -9 {p}')

