## cars: 

You have to download the images and the labels from: https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k
They are separate files, the images are in bdd100k, while the labels are in bdd100k_labels_release

Extract the following files from the downloaded zips:
	bdd100k/images/100k/val/
	bdd100k_labels_images_val.json

Place the files so that it looks like this:

	datasets/
	├─ car/
	│  ├─ bdd100k/
	│  │  ├─ val/
	│  │  ├─ bdd100k_labels_images_val.json


## lanes: 

https://www.kaggle.com/datasets/manideep1108/tusimple

Extract the following files from the downloaded zip:
	TUSimple/train_set/clips/0313-1
	TUSimple/train_set/clips/0313-2
	TUSimple/train_set/label_data_0313.json

Place the files so that it looks like this:

	datasets/
	├─ lanes/
	│  ├─ tusimple/
	│  │  ├─ train_set/
	│  │  │  ├─ clips/
	│  │  │  │  ├─ 0313-1/
	│  │  │  │  ├─ 0313-2/
	│  │  │  ├─ label_data_0313.json
