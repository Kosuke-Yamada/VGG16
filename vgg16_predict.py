from keras.models import model_from_json
import numpy as np
import os,random
from keras.preprocessing.image import img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from PIL import Image

def vgg16pred():

    batch_size=32
    test_dir='Data/test'
    display_dir='Data/display'
    file_name='vgg16_fine'
    label=['人','アニメキャラ','風景','食べ物','動物','ロゴ']#ラベルは自分で指定
    label.sort()

    #モデルと重みのロード
    json_string=open(file_name+'.json').read()
    model=model_from_json(json_string)
    model.load_weights(file_name+'.h5')
    
    model.compile(optimizer=SGD(lr=0.0001,momentum=0.9),
                loss='categorical_crossentropy',
                  metrics=['accuracy'])

    #テストデータ生成
    test_datagen=ImageDataGenerator(rescale=1.0/255)
    test_generator=test_datagen.flow_from_directory(
    test_dir,
    target_size=(224,224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
    )
    
    #モデル評価
    score=model.evaluate_generator(test_generator)
    print('\n test loss:',score[0])
    print('\n test_acc:',score[1])
    
    #モデルの可視化
    images=os.listdir(display_dir)
    img_predlist=[]
    for img in images:#読み込み画像
        temp_img=load_img(os.path.join(display_dir,img),target_size=(224,224))
        #画像の正規化
        temp_img_array=img_to_array(temp_img)
        temp_img_array=temp_img_array.astype('float')/255.0
        temp_img_array=temp_img_array.reshape((1,224,224,3))
        #画像の予測
        img_pred=model.predict(temp_img_array)
        #img_predlist.append(str(label[np.argmax(img_pred)]))
        print(str(img)+"\t"+str(label[np.argmax(img_pred)]))
        #画像の名前を変える用
        #os.rename(display_dir+"/"+img, display_dir+"/"+str(label[np.argmax(img_pred)])+"-"+str(img))

if __name__ == '__main__':
    vgg16pred()
