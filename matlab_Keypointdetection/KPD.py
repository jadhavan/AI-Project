import matlab.engine


# make sure in directory of the demo.m

def kpd():
    eng = matlab.engine.start_matlab('-nodisplay -nodesktop')
    n = eng.demo(nargout=0)
    eng.quit()
