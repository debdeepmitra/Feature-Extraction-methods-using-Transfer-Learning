from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator

def extract_feature_VGG16(path):

	#Pre-processing the imput image
	datagen = ImageDataGenerator()
	img_batch = datagen.flow_from_directory(
    	'path',
    	target_size=(224, 224),
    	batch_size=BATCH_SIZE,
    	color_mode='rgb',
	)

	#Generate the model with pretrained weights
	model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))

	for layer in model.layers:	#Don't train existing weights
  		layer.trainable = False

  	#Extracting features for the given image
  	features = model.predict(img_batch)

  	return features

def main():
	path = '/path/to/the/dir1'
	extracted_features = extract_feature_VGG16(path)

if __name__ == '__main__':
	main()