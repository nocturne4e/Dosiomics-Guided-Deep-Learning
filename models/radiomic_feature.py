from __future__ import print_function
import logging
import os
import six
import SimpleITK as sitk
import torch
import radiomics
from radiomics import featureextractor, getFeatureClasses
import numpy as np

settings = {}
settings['correctMask'] = True
settings['preCrop'] = True
settings['minimumROIDimensions'] = 1
settings['label'] = 1

# Get the PyRadiomics logger (default log-level = INFO)
logger = radiomics.logger
logger.setLevel(logging.ERROR)  # set level to DEBUG to include debug log messages in log file

extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
extractor.enableImageTypes(Original={})
extractor.addProvenance(False)

def extract_radiomic_features(images, masks):
	images = images[:, 0, ...] 
	(b, d, h, w) = images.shape
	#(b, d, h, w) = masks.shape
	radiomic_features = torch.FloatTensor()
	for i in range(b):
		img, mask = images[i], masks[i]
		#img = img.reshape((d, h, w))
		#mask = mask.reshape((d, h, w))
		img = sitk.GetImageFromArray(img)
		mask = sitk.GetImageFromArray(mask)
		featureVector = extractor.execute(img, mask)
		filter_featureVector = torch.from_numpy(np.array(list(featureVector.values())))
		radiomic_features = torch.cat((radiomic_features, filter_featureVector.reshape((1, -1))), 0)
	radiomic_features[torch.isnan(radiomic_features)] = 0

	return radiomic_features


def choose_thresholds(array,thre):
	thresholds = []
	for i in range(array.shape[0]):
		values = array[i].flatten()
		sorted_values = np.sort(values)[::-1]
		threshold = sorted_values[int(len(sorted_values) * thre)]
		thresholds.append(threshold)

	# 根据阈值将每个数组中的元素置为1或0
	binary_array = np.zeros_like(array)
	for i in range(array.shape[0]):
		binary_array[i] = (array[i] >= thresholds[i]).astype(int)
	return binary_array
