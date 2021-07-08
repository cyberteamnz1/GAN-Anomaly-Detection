### BiGAN model apply on NSL-KDD; accuracy: 91.12%
precision = 0.8727372840525844

recall = 0.9880776124055171

Accuracy = 0.9111958836053939

F1 = 0.9268328338571742

ROC_score = 0.953

## Notice: When load the model, discriminator and bigan need to be complied again due to the use of tensorflow.optimizer.

from tensorflow.keras.optimizers import Adam optimizer = Adam(0.001, 0.5)

from keras.models import load_model

encoder = load_model("BiGAN_encoder.hdf5")

bigan = load_model("BiGAN_bigan.hdf5")

discriminator = load_model("BiGAN_discriminator.hdf5")

discriminator.compile(loss=['binary_crossentropy'], optimizer=optimizer, metrics=['accuracy'])

bigan.compile(loss=['binary_crossentropy', 'binary_crossentropy'], optimizer=optimizer)
