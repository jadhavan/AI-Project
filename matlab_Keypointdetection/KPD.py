import matlab.engine


# make sure in directory of the demo.m
# depends on MATLAB, Image Processing Toolbox, Parallel Computing Toolbox,
# and MATLAB Distributed Computing Server

def kpd(filename):
    eng = matlab.engine.start_matlab('-nodisplay -nodesktop')
    #n = eng.demo(nargout=0)
    #nm = filename 'sample_img.png'
    x = eng.getKeyPoints(filename, nargout = 3)
    print x
    eng.quit()
