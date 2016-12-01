import matlab.engine
import json
import os
import timeit

# make sure in directory of the demo.m
# depends on MATLAB, Image Processing Toolbox, Parallel Computing Toolbox,
# and MATLAB Distributed Computing Server
# make take a JSON file make output to tensorflow files
# folder for frames with video associated
# try for reading youtube dataxs

def kpd(filename):
    eng = matlab.engine.start_matlab('-nodisplay -nodesktop')
    n = eng.demo(nargout=0)
    #nm = filename 'sample_img.png'
    x = eng.getKeyPoints(filename, nargout = 3)
    print x
    eng.quit()

def kpd_list(filelist, start=0, stop=-1, filename=None):
    
    eng = matlab.engine.start_matlab('-nodisplay -nodesktop')
    data = {}
    if stop == -1:
        fl = filelist[int(start):] # shifts slice to line up with indices
    else:
        fl = filelist[int(start):int(stop)+1] # shifts slice to like up with indices
    for f in fl:
        x, y, l = eng.getKeyPoints(f, nargout = 3)
        joints = {}
        for p in range(len(l)):
            joints[l[p]] = (x[0][p],y[0][p])  #nested lists for x and y but not l
        data[f] = joints
    eng.quit()
    
    #print data
    if filename:
        with open(filename, 'w+') as f:
            json.dump(data, f)
    return data

def getFileList():
    filelist = os.listdir("./frame")
    cwd = os.getcwd()
    framedir = os.path.join(cwd, "frame/")
    #print framedir
    filelist = [os.path.join(framedir,f) for f in filelist]
    return filelist
    

def runTest1():
    testfiles = ["sample_img.jpg", "sample_img.png", "Glenn_Standing_Left.jpg"]
    print timeit.timeit(lambda: kpd_list(testfiles, filename="testfile.json"), number = 2)

def runTest2():
    testfiles = ["0455_0205.jpg"]
    print testfiles
    kpd_list(testfiles, filename="testfile2.json")
    
def main():
    datafiles = getFileList()
    #print datafiles
    kpd_list(datafiles, 0, 10,  "framedata.json")
    return datafiles
