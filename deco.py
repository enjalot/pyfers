

def options(value='ttt'):
    print value
    def wrapper(target):
        print target.__name__
        def wrap(*args, **kwargs):
            print value
            print "args", args
            res = target(*args, **kwargs)
            return res
        return wrap
    return wrapper

@options('asdf')
def test(a, b):
    return a+b

@options()
def test2(a, b):
    return a+b

print test(1, 2)
print test2(1, 2)
