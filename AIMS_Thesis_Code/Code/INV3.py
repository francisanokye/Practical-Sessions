import time
import numpy as np
from sklearn.metrics import confusion_matrix
from keras.utils import plot_model
from keras import applications
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import SGD
import matplotlib.pyplot as plt

train_data = '/home/francisanokye/AIMS/Bitbucket/image-classification/trial_image_data/train'
validation_data = '/home/francisanokye/AIMS/Bitbucket/image-classification/trial_image_data/validation'

np.random.seed(0)
# dimensions of medical images.
img_width, img_height = 299, 299

# number of samples used for determining the samples_per_epoch
batch_size = 20

train_datagen = ImageDataGenerator(
    rescale=1./255,            # normalize pixel values to [0,1]
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(train_data, target_size=(img_height, img_width),
                                                    batch_size=batch_size, class_mode='categorical')

validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_directory(validation_data, target_size=(img_height, img_width),
                                                              batch_size=batch_size, class_mode='categorical')


"""
Inception-v3 is a convolutional neural network that is trained on more than a million images from the ImageNet database 
[1]. The network is 48 layers deep and can classify images into 1000 object categories, such as keyboard, mouse, pencil, 
and many animals.
"""
# USING THE INCEPTION-V3 BASE MODEL FROM KERAS WITHOUT THE TOP DENSE LAYERS WITH WEIGHTS FROM IMAGENET

base_model = applications.InceptionV3(
    weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# FREEZING ALL THE LAYERS

for layer in base_model.layers[:]:
    layer.trainable = False

# RESTORING THE GLOBAL AVERAGE POOLING LAYER AND BUILDING ON TOP LAST DENSE LAYER (CLASSIFIER)

x = base_model.output
x = GlobalAveragePooling2D(name='avg_pool')(x)
#x= Dense(256,activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(8, activation='softmax')(x)
V3model = Model(inputs=base_model.input, outputs=predictions)

V3model.compile(optimizer=optimizers.sgd(lr=2e-4, momentum=0.9,
                                         decay=0), loss='categorical_crossentropy', metrics=['accuracy'])
V3model.summary()

# TRAINING THE MODEL

V3_model = V3model.fit_generator(train_generator,
                                 steps_per_epoch=train_generator.samples // train_generator.batch_size,
                                 validation_data=validation_generator,
                                 validation_steps=validation_generator.samples // validation_generator.batch_size,
                                 epochs=25, verbose=1)

V3model.save('InceptionV3model.h5')
t = time.time()
print('Training time: %s' % (t - time.time()))

# Plot the accuracy and loss curves

acc = V3_model.history['acc']
val_acc = V3_model.history['val_acc']
loss = V3_model.history['loss']
val_loss = V3_model.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'g', label='Validation acc')
plt.title('Training and validation accuracy for InceptionV3')
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'g', label='Validation loss')
plt.title('Training and validation loss for InceptionV3')
plt.legend()
plt.show()

# Get the filenames from the generator
fnames = validation_generator.filenames

# Get the ground truth from generator
ground_truth = validation_generator.classes

# Get the label to class mapping from the generator
label2index = validation_generator.class_indices

# Getting the mapping from class index to class label
idx2label = dict((v, k) for k, v in label2index.items())

# Get the predictions from the model using the generator
predictions = V3model.predict_generator(
    validation_generator, steps=validation_generator.samples//validation_generator.batch_size, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)

# Show the errors
errors = np.where(predicted_classes != ground_truth)[0]
print("No of errors = {}/{}".format(len(errors), validation_generator.samples))
"""
for i in range(len(errors)):
    pred_class = np.argmax(predictions[errors[i]])
    pred_label = idx2label[pred_class]

    title = 'Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
        fnames[errors[i]].split('/')[0],
        pred_label,
        predictions[errors[i]][pred_class])

    original = load_img('{}/{}'.format(test_data, fnames[errors[i]]))
    plt.figure(figsize=[5, 5])
    plt.axis('off')
    plt.title(title)
    plt.imshow(original)
    plt.show()
"""
"""
    # To plot just a few of images.
    
     # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(errors[i].reshape(img_size, img_size, num_channels))
 """
# Show the correct predictions
correct = np.where(predicted_classes == ground_truth)[0]
print("No of correct predictions = {}/{}".format(len(correct),
                                                 validation_generator.samples))

"""
for i in range(len(correct)):
    pred_class = np.argmax(predictions[correct[i]])
    pred_label = idx2label[pred_class]

    title = 'Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
        fnames[correct[i]].split('/')[0],
        pred_label,
        predictions[correct[i]][pred_class])

    original = load_img('{}/{}'.format(test_data, fnames[correct[i]]))
    plt.figure(figsize=[5, 5])
    plt.axis('off')
    plt.title(title)
    plt.imshow(original)
    plt.show()

cm = confusion_matrix(ground_truth, predicted_classes)
cm

TruePositive = np.diag(cm)
print("TP is" + str(TruePositive))

FalsePositive = []
for i in range(12):
    FalsePositive.append(sum(cm[:, i]) - cm[i, i])

print("FP is" + str(FalsePositive))

FalseNegative = []
for i in range(12):
    FalseNegative.append(sum(cm[i, :]) - cm[i, i])

print("FN is" + str(FalseNegative))

TrueNegative = []
for i in range(12):
    temp = np.delete(cm, i, 0)   # delete ith row
    temp = np.delete(temp, i, 1)  # delete ith column
    TrueNegative.append(sum(sum(temp)))

print("TN is" + str(TrueNegative))

# To plot just a few of images.

"""
"""
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(errors[i].reshape(img_size, img_size, num_channels))



 def predict(model, img):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return preds[0]

img = image.load_img('test/Dog/110.jpg', target_size=(HEIGHT, WIDTH))
preds = predict(load_model(MODEL_FILE), img)       
"""
