# VIZ EDA
An exploratory data analysis tool to facilitate the visualisation of complex computer vision and object recognition datasets.  

## Run a new analysis
You will need image data and a COCO-dataset styled annotation file (.JSON)
1. Click "NEW ANALYSIS" on the home page.
![](assets/new_analysis.png)
2. Input the absolute path to your image data. For example: "/Users/me/project/data/val2017"
3. Upload your COCO-dataset styled JSON annotation file
![](assets/new_analysis_menu.png)
4. Click "ANALYSE". 
Analysing can take about 5-20 minutes, depending on the size of your data.

## Loading an existing analysis
You will need a feather file containing the results of a previous VIZ EDA analysis. This allows for quick visualization and exploratory data analysis without having to run the analysis again.

1. Click "EXISTING ANALISYS" on the home page.
![](assets/existing_analysis.png)
2. Upload your feather file
![](assets/existing_analysis_menu.png)
3. Click "VISUALISE"

## Batch analysis
The app can process multiple datasets where multiple annotations and images are merged before analysis. Your data
directory should look as follows:
```markdown
Example:
data
├── apple
│   ├── annotations
│   │   ├── instances.json
│   ├── images
│
├── banana
│   ├── annotations
│   │   ├── instances.json
│   ├── images
│
├── orange
│   ├── annotations
│   │   ├── instances.json
│   ├── images
│...
```
A temporary folder with the merged dataset (images and annotation) can be found in `./temp/results/merged`. The merged
JSON and image folder can be kept for future dataset analysis without batch processing.

## Exploratory data analysis

### Overview
This tab provides an overview of the dataset analysis.
![](assets/overview.png)

At the top of the page there are three tables:
1. The first table displays dataset information.
![](assets/info.png)
2. The second table shows a summary of the dataset:
  * Number of classes
  * Number of annotations
  * Number of images
  * Min., max., and avg. number of annotations per image
![](assets/summary.png)
3. The third table reports the analysis warnings: 
  * Whether the class distribution is uniform or not (i.e. whether there are classes that represent more than 80% of the dataset or less than 5%)
  * Whether there are images with no annotations
  * Whether there are annotations with no images
  * Whether there are images with wrong dimensions (assuming the Recycleye standard of 1920x1080)
  * Whether there are missing classes (i.e. are in "categories" but not in annotations)
  * Whether there are missing images (i.e. are referenced in annotations but are not in the folder provided to the app)
![](assets/warnings.png)

The rest of the page provides a summary of each class:
* Class name
* Class ID
* Number of annotations
* Percentage of annotations wrt total number of annotations in the dataset
* Number of images
* Percentage of images wrt total number of images in the dataset
* Number of unique images, i.e. where there are only annotations of that class
* Percentage of unique images wrt total number of images for that class
* 3 sample images of the class
![](assets/class.png)

### Objects per class
The bar graph displays the number of objects in the dataset provided, sorted by category. The pie chart shows a
breakdown of the proportion of objects in each category.
![](assets/objs_per_cat.png)

### Images per class
The bar graph displays the number of images containing one or more objects of a category. The pie chart shows a
breakdown of the images per category.
![](assets/imgs_per_cat.png)

### Objects per image
The first chart displays the average number of objects of a given class contained in image. For example, given images
with one or more cars, the chart displays the average number of cars in those images. To see the distribution of the
number of objects per image, click on a bin of your choice. To see the images that contain n objects of a given
category, on the probability histogram, click on a bin of your choice.
![](assets/objs_per_img.png)

### Proportion of object in image
The first chart displays the average proportion an object of a certain category takes up in an image. To see the
distribution of proportions an object takes up in image, click on a bin of your choice. To see the images that contain
objects of a given category taking up x% of a image, click on a bin of your choice.
![](assets/area_per_img.png)

## Anomaly detection
1. Select a category from the dropdown list to see the anomalies from that category. The anomaly objects are
highlighted in colour. Note that these objects were flagged by the model.
2. To manually flag an image containing an anomaly, simply click on the image.
3. Click "Export" to download an Excel file containing the list of manually flagged images, along with their respective
categories.

![](assets/anomalies.png)

banana anomaly             |  apple anomaly          |  orange anomaly
:-------------------------:|:-------------------------:|:-------------------------:
![](assets/banana.png)  |  ![](assets/apple.png) | ![](assets/orange.png)
