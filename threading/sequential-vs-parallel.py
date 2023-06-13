#!/usr/bin/env python3

import threading
import time

def sequential_run(name, counter):
   print ("  Starting " + name)
   do_work(name, counter)
   print ("  Exiting " + name)

class myThread (threading.Thread):
   def __init__(self, name, counter):
      threading.Thread.__init__(self)
      self.name = name
      self.counter = counter
   def run(self):
      print ("  Starting " + self.name)
      do_work(self.name, self.counter)
      print ("  Exiting " + self.name)

def do_work(name, counter):
   while counter:
      time.sleep(1)
      print ("    %s: %s" % (name, time.ctime(time.time())))
      counter -= 1

# Run sequentially
startTime = time.time()
sequential_run("Sequential-1", 2)
sequential_run("Sequential-2", 4)
print ("Sequential took: %s secs\n" % (time.time() - startTime))

# Create new threads
thread1 = myThread("Parallel-1", 2)
thread2 = myThread("Parallel-2", 4)

# Start new Threads
startTime = time.time()
thread1.start()
thread2.start()
thread1.join()
thread2.join()
print ("Parallel took: %s secs\n" % (time.time() - startTime))
