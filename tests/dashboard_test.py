import time
from app.layouts import serve_layout
from application import app
import pytest

app.layout = serve_layout

def test_dashboard001_summary(dash_duo):
    dash_duo.start_server(app)
    time.sleep(1)

    imagepath = dash_duo.driver.find_element_by_id("images-upload")
    imagepath.send_keys("/Users/xinwen/Desktop/val2017")
    upload = dash_duo.driver.find_element_by_xpath("//input[@type='file']")
    upload.send_keys("/Users/xinwen/Desktop/annotations/instances_val2017.json")
    time.sleep(1)
    analyse = dash_duo.driver.find_element_by_id("analyze-btn")
    analyse.click()
    time.sleep(12)

    h5title = ['Dataset name', 'Year', 'Number of images', 'Number of objects',\
               'Objects distribution', 'Images distribution', 'Number of classes',\
               'IDs range', 'Min bbox dimensions', 'Max bbox dimensions']
    elems = dash_duo.driver.find_elements_by_tag_name('h5')
    count = 0
    for elem in elems:
        count += 1
        time.sleep(1)
        assert elem.text in h5title
    assert count == len(h5title)


def test_dashboard002_toggle(dash_duo):
    dash_duo.start_server(app)
    time.sleep(1)

    imagepath = dash_duo.driver.find_element_by_id("images-upload")
    imagepath.send_keys("/Users/xinwen/Desktop/val2017")
    upload = dash_duo.driver.find_element_by_xpath("//input[@type='file']")
    upload.send_keys("/Users/xinwen/Desktop/annotations/instances_val2017.json")
    time.sleep(1)
    analyse = dash_duo.driver.find_element_by_id("analyze-btn")
    analyse.click()
    time.sleep(10)

    pops = ["minbbox", "maxbbox", "numim", "numob", "objectsDistribution",\
               "imageDistribution", "class", "idrange"]

    popheaders = ["Minimum Bounding Box", "Maximum Bounding Box", "Number of Images",\
                  "Number of Objects", "Distribution of Objects",\
                  "Distribution of Images", "Number of Classes", "Range of IDs"]
    count = 0
    for (pop,header) in zip(pops, popheaders):
        button = dash_duo.driver.find_element_by_id(f"popover-{pop}-target")
        button.click()
        time.sleep(1)
        h3 = dash_duo.driver.find_element_by_class_name("popover-header").text
        if h3 == header:
            count += 1
        button.click()
        time.sleep(1)

    assert count == len(popheaders)


if __name__ == "__main__":
    pytest.main(['--disable-warnings', 'dashboard_test.py'])