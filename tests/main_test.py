import time
from app.layouts import serve_layout
from application import app
from selenium.webdriver.common.keys import Keys
import pytest

app.layout = serve_layout


def test_entry_mainpage(dash_duo):

    dash_duo.start_server(app)
    time.sleep(1)
    pagetitle = dash_duo.driver.find_element_by_tag_name('h3').text
    assert pagetitle == 'New analysis'


def test_sidebar001_collapse(dash_duo):

    dash_duo.start_server(app)
    time.sleep(1)

    btn = dash_duo.driver.find_element_by_id('toggle')
    btn.click()
    time.sleep(1)
    sidebar = dash_duo.driver.find_element_by_id("sidebar")
    attribute = sidebar.get_attribute("class")
    assert attribute == "sidebar collapse"
    btn.click()
    time.sleep(1)
    attribute = sidebar.get_attribute("class")
    assert attribute == "sidebar"


def test_sidebar002_btns123(dash_duo):

    dash_duo.start_server(app)
    time.sleep(1)

    # other buttons in sidebar
    buttons = ['1', '3', '2']
    h3text = ['About vizEDA', 'Upload analysis', 'New analysis']
    for button, h3 in zip(buttons, h3text):
        time.sleep(1)
        dash_duo.driver.find_element_by_id(f"sidebar-btn-{button}").click()
        time.sleep(1)
        title = dash_duo.driver.find_element_by_tag_name('h3').text
        assert h3 == title
        time.sleep(2)

def test_callback001_newanalysis(dash_duo):

    dash_duo.start_server(app)
    time.sleep(2)

    # image path
    # default not valid and analyse button not enabled
    imagepath = dash_duo.driver.find_element_by_id("images-upload")
    valid = imagepath.get_attribute("class")
    assert 'is-valid' not in valid
    fileupload = dash_duo.driver.find_element_by_id("upload-btn")
    style = fileupload.get_attribute("style")
    assert 'green' not in style
    analyse = dash_duo.driver.find_element_by_id("analyze-btn")
    assert analyse.is_enabled() is False

    # existent path
    imagepath.send_keys("/Users")
    valid = imagepath.get_attribute("class")
    assert 'is-valid' in valid
    assert analyse.is_enabled() is False
    # incorrect path
    imagepath.send_keys("/nonexistent")
    valid = imagepath.get_attribute("class")
    assert 'is-valid' not in valid
    assert analyse.is_enabled() is False
    # correct path
    while imagepath.get_attribute("value") != '':
        imagepath.send_keys(Keys.BACKSPACE)
    imagepath.send_keys("/Users/xinwen/Desktop/val2017")
    valid = imagepath.get_attribute("class")
    assert 'is-valid' in valid
    assert analyse.is_enabled() is False

    upload = dash_duo.driver.find_element_by_xpath("//input[@type='file']")
    upload.send_keys("/Users/xinwen/Desktop/annotations/instances_val2017.json")
    time.sleep(2)
    style = fileupload.get_attribute("style")
    assert 'green' in style
    assert analyse.is_enabled()


def test_sidebar003_btns45678(dash_duo):
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

    buttons = ['5', '4', '6', '7', '8']
    h3text = ['Warnings', 'Dashboard', 'Classes', 'Stats', 'Anomalies']
    for button, h3 in zip(buttons, h3text):
        time.sleep(1)
        dash_duo.driver.find_element_by_id(f"sidebar-btn-{button}").click()
        time.sleep(3)
        if button != '8':
            title = dash_duo.driver.find_element_by_tag_name('h3').text
            time.sleep(1)
            assert h3 == title
        else:
            select = dash_duo.driver.find_element_by_class_name('Select-placeholder').text
            assert select == 'Select an algorithm'


if __name__ == "__main__":
    pytest.main(['--disable-warnings', 'main_test.py'])