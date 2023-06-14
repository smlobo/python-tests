#!/usr/bin/env python3

import threading
import time

# Threads:
#   Sequential run of a function
#   Parallel run of the same function using threads
#   Prints summary timing

# Function which does the sequential run
def sequential_run(name, counter):
   print ("  Starting " + name)
   do_work(name, counter)
   print ("  Exiting " + name)

# Class that does the parallel run
class myThread (threading.Thread):
   def __init__(self, name, counter):
      threading.Thread.__init__(self)
      self.name = name
      self.counter = counter
   def run(self):
      print ("  Starting " + self.name)
      do_work(self.name, self.counter)
      print ("  Exiting " + self.name)

# Common worker function that sleeps for given seconds
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

# Start new Threads in parallel
startTime = time.time()
thread1.start()
thread2.start()
thread1.join()
thread2.join()
print ("Parallel took: %s secs\n" % (time.time() - startTime))
