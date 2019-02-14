from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D,Input
from keras.applications.vgg16 import VGG16#VGG16モデル
from keras.preprocessing.image import ImageDataGenerator#データ拡張を行うクラス
from keras.optimizers import SGD
from keras.callbacks import CSVLogger

n_categories = 6#カテゴリの数
batch_size = 32#バッチ数
train_dir = 'Data/train/'#学習データのディレクトリ
validation_dir = 'Data/validation/'#交差検証を行うデータのディレクトリ
file_name = 'vgg16_fine'#ファインチューニングを行った後のファイル名

#VGG16を呼び出す(重み→imagenet，全結合層を使わない→include_p = false，入力画像設定shape(縦,横,RGB))
base_model = VGG16(weights = 'imagenet', include_top = False, input_tensor = Input(shape = (224,224,3)))

#VGG16に新しい全結合層を取り付ける
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024,activation = 'relu')(x)
prediction = Dense(n_categories,activation = 'softmax')(x)
model = Model(inputs = base_model.input,outputs = prediction)

#14層までの重みを更新しない
for layer in base_model.layers[:15]:
    layer.trainable = False

model.compile(optimizer = SGD(lr = 0.0001,momentum = 0.9),
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

model.summary()

json_string = model.to_json()
open(file_name+'.json','w').write(json_string)

#以下画像の前処理
train_datagen = ImageDataGenerator(
    rescale = 1.0/255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True)

validation_datagen = ImageDataGenerator(rescale = 1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size = (224,224),
    batch_size = batch_size,
    class_mode = 'categorical',
    shuffle = True
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size = (224,224),
    batch_size = batch_size,
    class_mode = 'categorical',
    shuffle = True
)

hist = model.fit_generator(train_generator,
                         epochs = 100,
                         verbose = 1,
                         validation_data = validation_generator,
                         callbacks = [CSVLogger(file_name+'.csv')])

#save weights
model.save(file_name+'.h5')
