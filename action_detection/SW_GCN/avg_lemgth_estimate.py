import os
import numpy as np
from bvh import Bvh
import subprocess
import moviepy.editor
import statistics


class AvgLengthEstimate:
    """
    calculates average action length over whole dataset (InHARD)
    """

    def __init__(self):

        # path to data
        self.cutoff = 100
        self.path_to_folder = ""  # ToDO: insert path to training data (InHARD)
        self.lengthes = []
        self.avgs = []

    def avgL(self):
        counter = 0
        # iterate over folders
        for folder in os.listdir(self.path_to_folder):
            if folder.endswith(".txt"):
                continue
            if folder.endswith("No action"):
                continue
            print(str(folder))
            # iterate over bvh files
            for filename in os.listdir(self.path_to_folder + folder):
                counter = counter + 1
                if counter > self.cutoff:
                    counter = 0
                    break
                fp = self.get_length(self.path_to_folder + folder + "\\" + filename)
                if fp < 0:
                    continue
                self.lengthes.append(fp)
                print("progress " + str(counter))
            if len(self.lengthes) == 0:
                continue
            self.avgs.append(statistics.mean(self.lengthes))
            self.lengthes.clear()

    def save(self):
        with open('', 'w') as f:  # ToDO: insert path to store result
            for element in self.avgs:
                f.write(str(element) + " " + str(element / 120) + "s")
                f.write("\n")

    def get_length(self, filename):
        with open(filename) as f:
            try:
                fp = Bvh(f.read()).nframes
            except:
                return -1
            return fp


if __name__ == '__main__':
    instance = AvgLengthEstimate()
    # instance.load_bvh()
    instance.avgL()
    instance.save()
