import os

import random

li = open('train.txt','r').readlines()
random.shuffle(li)
print len(li)
#train = li
valid = li[:10000]
train = li[10000:]

print len(train)
print len(valid)
f = open('train.txt','w')
f.write(''.join(train))
f.close()

f = open('valid.txt','w')
f.write(''.join(valid))
f.close()
