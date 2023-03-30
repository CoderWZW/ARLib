import random
train = []
test = []
val=[]
with open('ratings.dat') as f:
    for line in f:
        items = line.strip().split('::')
        new_line = ' '.join(items[:-1])+'\n'
        if int(items[-2])<4:
            continue
        num=random.random()
        if num > 0.2:
            train.append(new_line)
        elif num > 0.1:
            val.append(new_line)
        else:
            test.append(new_line)

with open('train.txt','w') as f:
    f.writelines(train)

with open('test.txt','w') as f:
    f.writelines(test)

with open('val.txt','w') as f:
    f.writelines(val)
