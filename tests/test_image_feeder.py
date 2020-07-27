from mouseyes.input_feeder import InputFeeder

def image():
    ifeed = InputFeeder('image')
    ifeed.load_data()
    assert ifeed.cap is not None