from __future__ import print_function
import time
import timeit
import datetime

def get_datetime_seconds():
    return (datetime.datetime.utcnow()-datetime.datetime(1970,1,1)).total_seconds()

class Timer(object):
    def __init__(self, silent=False):
        self.cumulative_secs = {}
        self.current_ticks = {}
        self.silent = silent

    def tick(self, name='default'):
        if name not in self.current_ticks:
            self.current_ticks[name] = time.time()

            return 0.0
        else:
            if name not in self.cumulative_secs:
                self.cumulative_secs[name] = 0
            t = time.time()
            self.cumulative_secs[name] += t - self.current_ticks[name]
            self.current_ticks.pop(name)

            return self.cumulative_secs[name]

    def tock(self, name='default'):
        self.tick(name)
        value = self.cumulative_secs[name]
        if not self.silent:
            print('Time taken for {0}: {1:.4f}s'.format(name, value))
        self.cumulative_secs.pop(name)
        self.current_ticks.pop(name, None)

        return value
