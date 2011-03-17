import time

class Timing(object):
    def __init__(self):
        self.timings = {}
        self.col = self.__collector()
        self.col.next()                 #coroutine syntax

    def __collector(self):
        while True:
            (name, t) = (yield)         #coroutine syntax
            if name in self.timings:
                self.timings[name]["timings"] += [t]
                self.timings[name]["count"] += 1
                self.timings[name]["total"] += t
            else:
                self.timings[name] = {} #if this entry doesn't exist yet
                self.timings[name]["timings"] = [t]
                self.timings[name]["count"] = 1
                self.timings[name]["total"] = t

    def __call__(self, pretty_name=None):
        """Turn the object into a decorator"""
        def wrap(func):                         #the decorator
            def wrapped(*args, **kwargs):
                t1 = time.time()                #start time
                res = func(*args, **kwargs)     #call the originating function
                t2 = time.time()                #stop time
                t = (t2-t1)*1000.0              #time in milliseconds

                if pretty_name is not None:     #allow user to choose timer name
                    name = pretty_name
                else:
                    name = func.__name__

                data = (name, t)
                self.col.send(data)             #collect the data
                return res 
            return wrapped
        return wrap


    def __str__(self):
        s = "Timings:\n"
        #print dir(self)
        sk = sorted(self.timings.items(), key=lambda x : x[1]["total"])
        #for key in sorted(self.timings.keys()):
        for t in sk:
            s += "%-20s | " % t[0] 
            ts = t[1]["timings"]
            count = t[1]["count"]
            total = t[1]["total"]
            s += "average: %.8f | total: %.8f | count: %d\n" % (total / count, total, count)
        return "%s" % s 

timings = Timing()


class SimpleTiming(object):
    def __init__(self):
        self.timings = {}

    def __call__(self, func):
        def wrapped(*args, **kwargs):
            t1 = time.time()                #start time
            res = func(*args, **kwargs)     #call the originating function
            t2 = time.time()                #stop time
            t = (t2-t1)*1000.0              #time in milliseconds
            self.timings[func.__name__] = t
            return res
        return wrapped

    def __str__(self):
        return "%s" % self.timings




if __name__ == "__main__":
    

    stimings = SimpleTiming()

    @stimings
    def add(x,y):
        for i in range(10000):
            c = x + y
        return c

    @stimings()
    def multiply(x,y):
        for i in range(10000):
            c = x * y
        return c

    timings = Timing()
    @timings("Add")
    def add1(a,b):
        for i in range(10000):
            c = a + b
        return c

    @timings()
    def multiply1(a,b):
        for i in range(10000):
            c = a * b
        return c


    for i in range(100):
        add(3.,4.)
        multiply(3., 4.)
        add1(3., 4.)
        multiply1(3., 4.)


    print timings
    print stimings
