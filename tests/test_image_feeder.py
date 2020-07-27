from mouseyes.input_feeder import InputFeeder

def test_image():
    TEST_IMAGE='tests/resources/center.jpg'

    ifeed = InputFeeder(TEST_IMAGE)
    assert ifeed.input_type == 'image'
    assert len(ifeed) == 1
    frames = 0
    for img in ifeed:
        assert img is not None
        assert img.shape == (297, 480, 3)
        frames+=1
    assert frames == 1

def test_video():
    TEST_VIDEO='tests/resources/elon.mp4'
    VIDEO_LEN=133
    VIDEO_SIZE=(576, 1024, 3)

    ifeed = InputFeeder(TEST_VIDEO)
    assert ifeed.input_type == 'video'
    assert len(ifeed) == VIDEO_LEN
    frames = 0
    for img in ifeed:
        assert img is not None
        assert img.shape == VIDEO_SIZE
        frames+=1
    assert frames+1 == VIDEO_LEN
