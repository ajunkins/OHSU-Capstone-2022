
#import sys
#sys.path.insert(0, '/home/ohsu/Downloads/jetson-inference/python/training/classification')
import mv_classification
import time

mv_classification.printTest()

classifier = mv_classification.MachineVision()
result, confidence = classifier.classifyImage()
print("\nMachineVision.classifyImage returned result: '{:s}' and confidence: {:f}%".format(result, confidence))
t0 = time.clock()
print(t0)
