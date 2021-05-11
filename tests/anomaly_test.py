import time
from app.layouts import serve_layout
from application import app
from selenium.webdriver.common.keys import Keys
import pytest

app.layout = serve_layout

h5title = ['Objects distribution', 'Algorithm', 'Number of Anomalous Images',\
               'Number of Anomalous Object', 'Class with Highest Number of Anomalies']

somecol = ['image_id', 'id', 'cat_id', 'cat_name']

def test_anomaly001_size(dash_duo):
    dash_duo.start_server(app)
    time.sleep(1)

    dash_duo.driver.find_element_by_id("images-upload").send_keys("/Users/xinwen/Desktop/VOC_COCO/images")
    dash_duo.driver.find_element_by_xpath("//input[@type='file']").send_keys("/Users/xinwen/Desktop/VOC_COCO/annotations/voc_add_anomaly.json")
    time.sleep(1)
    dash_duo.driver.find_element_by_id("analyze-btn").analyse.click()
    time.sleep(8)
    dash_duo.driver.find_element_by_id('sidebar-btn-8').click()
    time.sleep(1)
    dash_duo.driver.find_element_by_class_name("Select-placeholder").click()
    time.sleep(1)
    dash_duo.driver.find_element_by_css_selector("div.Select-control input").send_keys("Object Size" + Keys.ENTER)
    time.sleep(1)
    dash_duo.driver.find_element_by_id('update-button').click()
    time.sleep(2)
    elems = dash_duo.driver.find_elements_by_tag_name('h5')
    count = 0
    for elem in elems:
        count += 1
        time.sleep(1)
        assert elem.text in h5title
    assert count == len(h5title)

    button = dash_duo.driver.find_element_by_id('table-toggle-object_size')
    button.click()
    time.sleep(1)
    #look for table
    dash_duo.driver.find_element_by_class_name('cell-table')
    time.sleep(2)

    for i in range(4):
        col = dash_duo.driver.find_element_by_css_selector(f'th.dash-header.column-{i}')
        name = col.get_attribute("data-dash-column")
        assert name in somecol
        time.sleep(1)

    button.click()
    time.sleep(1)


def test_anomaly002_autoencoder(dash_duo):
    dash_duo.start_server(app)
    time.sleep(1)
    dash_duo.driver.find_element_by_id("images-upload").send_keys("/Users/xinwen/Desktop/VOC_COCO/images")
    dash_duo.driver.find_element_by_xpath("//input[@type='file']").send_keys(
        "/Users/xinwen/Desktop/VOC_COCO/annotations/voc_add_anomaly.json")
    time.sleep(1)
    dash_duo.driver.find_element_by_id("analyze-btn").analyse.click()
    time.sleep(8)
    dash_duo.driver.find_element_by_id('sidebar-btn-8').click()
    time.sleep(1)
    dash_duo.driver.find_element_by_class_name("Select-placeholder").click()
    time.sleep(1)
    dash_duo.driver.find_element_by_css_selector("div.Select-control input").send_keys("Autoencoder" + Keys.ENTER)
    time.sleep(1)
    dash_duo.driver.find_element_by_id('update-button').click()
    time.sleep(2)
    elems = dash_duo.driver.find_elements_by_tag_name('h5')
    count = 0
    for elem in elems:
        count += 1
        time.sleep(1)
        assert elem.text in h5title
    assert count == len(h5title)

    button = dash_duo.driver.find_element_by_id('table-toggle-autoencoder')
    button.click()
    time.sleep(1)
    # look for table
    dash_duo.driver.find_element_by_class_name('cell-table')
    time.sleep(2)

    for i in range(4):
        col = dash_duo.driver.find_element_by_css_selector(f'th.dash-header.column-{i}')
        name = col.get_attribute("data-dash-column")
        assert name in somecol
        time.sleep(1)

    button.click()
    time.sleep(1)


def test_anomaly003_manualfeaturelof(dash_duo):
    dash_duo.start_server(app)
    time.sleep(1)
    dash_duo.driver.find_element_by_id("images-upload").send_keys("/Users/xinwen/Desktop/VOC_COCO/images")
    dash_duo.driver.find_element_by_xpath("//input[@type='file']").send_keys(
        "/Users/xinwen/Desktop/VOC_COCO/annotations/voc_add_anomaly.json")
    time.sleep(1)
    dash_duo.driver.find_element_by_id("analyze-btn").analyse.click()
    time.sleep(8)
    dash_duo.driver.find_element_by_id('sidebar-btn-8').click()
    time.sleep(1)
    dash_duo.driver.find_element_by_class_name("Select-placeholder").click()
    time.sleep(1)
    dash_duo.driver.find_element_by_css_selector("div.Select-control input").send_keys("LOF-Manually Extracted Feature" + Keys.ENTER)
    time.sleep(1)
    dash_duo.driver.find_element_by_id('update-button').click()
    time.sleep(2)
    elems = dash_duo.driver.find_elements_by_tag_name('h5')
    count = 0
    for elem in elems:
        count += 1
        time.sleep(1)
        assert elem.text in h5title
    assert count == len(h5title)

    button = dash_duo.driver.find_element_by_id('table-toggle-manual_feature_lof')
    button.click()
    time.sleep(1)
    # look for table
    dash_duo.driver.find_element_by_class_name('cell-table')
    time.sleep(2)

    for i in range(4):
        col = dash_duo.driver.find_element_by_css_selector(f'th.dash-header.column-{i}')
        name = col.get_attribute("data-dash-column")
        assert name in somecol
        time.sleep(1)

    button.click()
    time.sleep(1)


def test_anomaly004_manualfeatureiforest(dash_duo):
    dash_duo.start_server(app)
    time.sleep(1)
    dash_duo.driver.find_element_by_id("images-upload").send_keys("/Users/xinwen/Desktop/VOC_COCO/images")
    dash_duo.driver.find_element_by_xpath("//input[@type='file']").send_keys(
        "/Users/xinwen/Desktop/VOC_COCO/annotations/voc_add_anomaly.json")
    time.sleep(1)
    dash_duo.driver.find_element_by_id("analyze-btn").analyse.click()
    time.sleep(8)
    dash_duo.driver.find_element_by_id('sidebar-btn-8').click()
    time.sleep(1)
    dash_duo.driver.find_element_by_class_name("Select-placeholder").click()
    time.sleep(1)
    dash_duo.driver.find_element_by_css_selector("div.Select-control input").send_keys("iForest-Manually Extracted Feature" + Keys.ENTER)
    time.sleep(1)
    dash_duo.driver.find_element_by_id('update-button').click()
    time.sleep(2)
    elems = dash_duo.driver.find_elements_by_tag_name('h5')
    count = 0
    for elem in elems:
        count += 1
        time.sleep(1)
        assert elem.text in h5title
    assert count == len(h5title)

    button = dash_duo.driver.find_element_by_id('table-toggle-manual_feature_iforest')
    button.click()
    time.sleep(1)
    # look for table
    dash_duo.driver.find_element_by_class_name('cell-table')
    time.sleep(2)

    for i in range(4):
        col = dash_duo.driver.find_element_by_css_selector(f'th.dash-header.column-{i}')
        name = col.get_attribute("data-dash-column")
        assert name in somecol
        time.sleep(1)

    button.click()
    time.sleep(1)


def test_anomaly005_hoglof(dash_duo):
    dash_duo.start_server(app)
    time.sleep(1)
    dash_duo.driver.find_element_by_id("images-upload").send_keys("/Users/xinwen/Desktop/VOC_COCO/images")
    dash_duo.driver.find_element_by_xpath("//input[@type='file']").send_keys(
        "/Users/xinwen/Desktop/VOC_COCO/annotations/voc_add_anomaly.json")
    time.sleep(1)
    dash_duo.driver.find_element_by_id("analyze-btn").analyse.click()
    time.sleep(8)
    dash_duo.driver.find_element_by_id('sidebar-btn-8').click()
    time.sleep(1)
    dash_duo.driver.find_element_by_class_name("Select-placeholder").click()
    time.sleep(1)
    dash_duo.driver.find_element_by_css_selector("div.Select-control input").send_keys("LOF-HOG" + Keys.ENTER)
    time.sleep(1)
    dash_duo.driver.find_element_by_id('update-button').click()
    time.sleep(2)
    elems = dash_duo.driver.find_elements_by_tag_name('h5')
    count = 0
    for elem in elems:
        count += 1
        time.sleep(1)
        assert elem.text in h5title
    assert count == len(h5title)

    button = dash_duo.driver.find_element_by_id('table-toggle-hog_lof')
    button.click()
    time.sleep(1)
    # look for table
    dash_duo.driver.find_element_by_class_name('cell-table')
    time.sleep(2)

    for i in range(4):
        col = dash_duo.driver.find_element_by_css_selector(f'th.dash-header.column-{i}')
        name = col.get_attribute("data-dash-column")
        assert name in somecol
        time.sleep(1)

    button.click()
    time.sleep(1)


def test_anomaly006_hogiforest(dash_duo):
    dash_duo.start_server(app)
    time.sleep(1)
    dash_duo.driver.find_element_by_id("images-upload").send_keys("/Users/xinwen/Desktop/VOC_COCO/images")
    dash_duo.driver.find_element_by_xpath("//input[@type='file']").send_keys(
        "/Users/xinwen/Desktop/VOC_COCO/annotations/voc_add_anomaly.json")
    time.sleep(1)
    dash_duo.driver.find_element_by_id("analyze-btn").analyse.click()
    time.sleep(8)
    dash_duo.driver.find_element_by_id('sidebar-btn-8').click()
    time.sleep(1)
    dash_duo.driver.find_element_by_class_name("Select-placeholder").click()
    time.sleep(1)
    dash_duo.driver.find_element_by_css_selector("div.Select-control input").send_keys("iForest-HOG" + Keys.ENTER)
    time.sleep(1)
    dash_duo.driver.find_element_by_id('update-button').click()
    time.sleep(2)
    elems = dash_duo.driver.find_elements_by_tag_name('h5')
    count = 0
    for elem in elems:
        count += 1
        time.sleep(1)
        assert elem.text in h5title
    assert count == len(h5title)

    button = dash_duo.driver.find_element_by_id('table-toggle-hog_iforest')
    button.click()
    time.sleep(1)
    # look for table
    dash_duo.driver.find_element_by_class_name('cell-table')
    time.sleep(2)

    for i in range(4):
        col = dash_duo.driver.find_element_by_css_selector(f'th.dash-header.column-{i}')
        name = col.get_attribute("data-dash-column")
        assert name in somecol
        time.sleep(1)

    button.click()
    time.sleep(1)


def test_anomaly007_cnnlof(dash_duo):
    dash_duo.start_server(app)
    time.sleep(1)
    dash_duo.driver.find_element_by_id("images-upload").send_keys("/Users/xinwen/Desktop/VOC_COCO/images")
    dash_duo.driver.find_element_by_xpath("//input[@type='file']").send_keys(
        "/Users/xinwen/Desktop/VOC_COCO/annotations/voc_add_anomaly.json")
    time.sleep(1)
    dash_duo.driver.find_element_by_id("analyze-btn").analyse.click()
    time.sleep(8)
    dash_duo.driver.find_element_by_id('sidebar-btn-8').click()
    time.sleep(1)
    dash_duo.driver.find_element_by_class_name("Select-placeholder").click()
    time.sleep(1)
    dash_duo.driver.find_element_by_css_selector("div.Select-control input").send_keys("LOF-CNN" + Keys.ENTER)
    time.sleep(1)
    dash_duo.driver.find_element_by_id('update-button').click()
    time.sleep(2)
    elems = dash_duo.driver.find_elements_by_tag_name('h5')
    count = 0
    for elem in elems:
        count += 1
        time.sleep(1)
        assert elem.text in h5title
    assert count == len(h5title)

    button = dash_duo.driver.find_element_by_id('table-toggle-cnn_lof')
    button.click()
    time.sleep(1)
    # look for table
    dash_duo.driver.find_element_by_class_name('cell-table')
    time.sleep(2)

    for i in range(4):
        col = dash_duo.driver.find_element_by_css_selector(f'th.dash-header.column-{i}')
        name = col.get_attribute("data-dash-column")
        assert name in somecol
        time.sleep(1)

    button.click()
    time.sleep(1)


def test_anomaly008_cnniforest(dash_duo):
    dash_duo.start_server(app)
    time.sleep(1)
    dash_duo.driver.find_element_by_id("images-upload").send_keys("/Users/xinwen/Desktop/VOC_COCO/images")
    dash_duo.driver.find_element_by_xpath("//input[@type='file']").send_keys(
        "/Users/xinwen/Desktop/VOC_COCO/annotations/voc_add_anomaly.json")
    time.sleep(1)
    dash_duo.driver.find_element_by_id("analyze-btn").analyse.click()
    time.sleep(8)
    dash_duo.driver.find_element_by_id('sidebar-btn-8').click()
    time.sleep(1)
    dash_duo.driver.find_element_by_class_name("Select-placeholder").click()
    time.sleep(1)
    dash_duo.driver.find_element_by_css_selector("div.Select-control input").send_keys("iForest-CNN" + Keys.ENTER)
    time.sleep(1)
    dash_duo.driver.find_element_by_id('update-button').click()
    time.sleep(2)
    elems = dash_duo.driver.find_elements_by_tag_name('h5')
    count = 0
    for elem in elems:
        count += 1
        time.sleep(1)
        assert elem.text in h5title
    assert count == len(h5title)

    button = dash_duo.driver.find_element_by_id('table-toggle-cnn_iforest')
    button.click()
    time.sleep(1)
    # look for table
    dash_duo.driver.find_element_by_class_name('cell-table')
    time.sleep(2)

    for i in range(4):
        col = dash_duo.driver.find_element_by_css_selector(f'th.dash-header.column-{i}')
        name = col.get_attribute("data-dash-column")
        assert name in somecol
        time.sleep(1)

    button.click()
    time.sleep(1)


def test_anomaly009_imageai(dash_duo):
    dash_duo.start_server(app)
    time.sleep(1)
    dash_duo.driver.find_element_by_id("images-upload").send_keys("/Users/xinwen/Desktop/VOC_COCO/images")
    dash_duo.driver.find_element_by_xpath("//input[@type='file']").send_keys(
        "/Users/xinwen/Desktop/VOC_COCO/annotations/voc_add_anomaly.json")
    time.sleep(1)
    dash_duo.driver.find_element_by_id("analyze-btn").analyse.click()
    time.sleep(8)
    dash_duo.driver.find_element_by_id('sidebar-btn-8').click()
    time.sleep(1)
    dash_duo.driver.find_element_by_class_name("Select-placeholder").click()
    time.sleep(1)
    dash_duo.driver.find_element_by_css_selector("div.Select-control input").send_keys("ImageAI" + Keys.ENTER)
    time.sleep(1)
    dash_duo.driver.find_element_by_id('update-button').click()
    time.sleep(2)
    elems = dash_duo.driver.find_elements_by_tag_name('h5')
    count = 0
    for elem in elems:
        count += 1
        time.sleep(1)
        assert elem.text in h5title
    assert count == len(h5title)

    button = dash_duo.driver.find_element_by_id('table-toggle-imageai')
    button.click()
    time.sleep(1)
    dash_duo.driver.find_element_by_class_name('cell-table')
    time.sleep(2)

    for i in range(4):
        col = dash_duo.driver.find_element_by_css_selector(f'th.dash-header.column-{i}')
        name = col.get_attribute("data-dash-column")
        assert name in somecol
        time.sleep(1)

    button.click()
    time.sleep(1)

