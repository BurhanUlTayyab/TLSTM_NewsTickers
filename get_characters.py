#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import codecs

DIGITS = "~!%'()+,-.\/0123456789:ABCDEFGIJKLMNOPRSTUVWYabcdefghiklmnoprstuvwxz-V،د‘“ ؤب,گ0ذصط3وLِbT2dh9ٰٴxAڈlژ؛؟أGاpث4/س7ًtCهKیُS\"۔WOcgk…ٓosw(ﷺجڑ.آئکتخز6غEشہقنضDNR8ظ:fnrvzپچB’”لء%)ْFحر5عںھف!JمIM#ّےUYَae'Pimة1uٹ+".decode('utf-8')
char = {}

files1 = open('train.txt','r').readlines()
files2 = open('valid.txt','r').readlines()
files = []
for x in files1:
    files.append(x.strip('\n'))

for x in files2:
    files.append(x.strip('\n'))



for f in files:
    f1=codecs.open(os.path.splitext(f)[0]+ '.gt.txt','r',encoding='utf8')
    while True:
        c = f1.read(1)
        if not c:
            break
        char[c] = 1
    f1.close()


print len(char)
l = char.keys()
print len(DIGITS)
print len(set(l))
x = set(l)
x = filter(lambda a:a != '\n' , x)
#x = ''.join(l).strip().split()
for y in x:
    if(DIGITS.find(y) == -1):
        print y
        DIGITS+=y
print len(x)
print len(DIGITS)
d = DIGITS.split()
d = set(d)
d = ''.join(d)
print len(d)
g = codecs.open('ch.txt','w','utf-8')
g.write(d)
g.close()
