#Leo: example taken from 
#https://stackoverflow.com/questions/285061/how-do-you-programmatically-set-an-attribute
class X():
    def __init__(self, string):
        self.dummy = 0
        setattr(self, string, 'magic')
        
x = X('leo')
print(x.leo)
