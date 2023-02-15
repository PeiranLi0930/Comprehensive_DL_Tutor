result = [1,2,1,2,3,3,3]
freq = 0
num = result[0]
for i in result:
    if (result.count(i) > freq):
        freq = result.count(i)
        num = i
    elif (result.count(i) == freq and i < num):
        num = i


print(num)