import random
train = []
test = []
val=[]
with open('ratings.txt') as f:
    for line in f:
        num=random.random()
        if num < 0.6:
            train.append(line)
        elif num > 0.8:
            val.append(line)
        else:
            test.append(line)

with open('train.txt','w') as f:
    f.writelines(train)

with open('test.txt','w') as f:
    f.writelines(test)

with open('val.txt','w') as f:
    f.writelines(val)
