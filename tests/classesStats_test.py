import time
from app.layouts import serve_layout
from application import app
import pytest
from selenium.webdriver.common.keys import Keys
import numpy as np

app.layout = serve_layout

def test_stats001_stats(dash_duo):
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
    dash_duo.driver.find_element_by_id('sidebar-btn-7').click()
    time.sleep(3)
    h5title = ['Objects Distribution', 'Images Distribution',\
               'Number and Size of Objects (Unit: thousand pixels)',\
               'Average Size of Objects', 'Objects Size Distribution']
    elems = dash_duo.driver.find_elements_by_class_name('card-title')
    time.sleep(1)
    for elem in elems:
        assert elem.text in h5title

    graphs = dash_duo.driver.find_elements_by_class_name('dash-graph')
    assert len(graphs) == 7



def test_classes001_class(dash_duo):
    dash_duo.start_server(app)
    time.sleep(1)

    imagepath = dash_duo.driver.find_element_by_id("images-upload")
    imagepath.send_keys("/Users/xinwen/Desktop/VOC_COCO/images")
    upload = dash_duo.driver.find_element_by_xpath("//input[@type='file']")
    upload.send_keys("/Users/xinwen/Desktop/VOC_COCO/annotations/voc_add_anomaly.json")
    time.sleep(1)
    analyse = dash_duo.driver.find_element_by_id("analyze-btn")
    analyse.click()
    time.sleep(8)
    dash_duo.driver.find_element_by_id('sidebar-btn-6').click()
    time.sleep(1)
    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",\
               "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",\
               "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    h5title = ['Number of images', 'Number of objects', 'Min bbox width', 'Min bbox height', \
               'Max bbox width', 'Max bbox height', 'Average Size']
    data = []
    select = dash_duo.driver.find_element_by_class_name("Select-placeholder")
    select.click()
    time.sleep(1)
    for i,cls in enumerate(classes):
        dash_duo.driver.find_element_by_css_selector("div.Select-control input").send_keys(cls+Keys.ENTER)
        time.sleep(1)
        elems = dash_duo.driver.find_elements_by_tag_name('h5')
        tmp = []
        for j,elem in enumerate(elems):
            time.sleep(1)
            assert elem.text in h5title
        tmpdata = dash_duo.driver.find_elements_by_tag_name('h4')
        for t in tmpdata:
            tmp.append(t.text)
        data.append(tmp)
    np.savetxt("./tests/Data.csv", data, delimiter=", ", fmt='% s')


if __name__ == "__main__":
    pytest.main(['--disable-warnings', 'classesStats_test.py'])