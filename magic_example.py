#Leo: example taken from 
#https://stackoverflow.com/questions/285061/how-do-you-programmatically-set-an-attribute
#to modify Python code dynamically
class X():
    def __init__(self, dic):
        self.dummy = 0
        for key in dic.keys():
            setattr(self, key, dic[key])

gridDictionary={'a':9,'b':6,'c':3}

x = X(gridDictionary)
print(x.a)
print(x.b)
print(x.c)
