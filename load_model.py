from keras.models import load_model
import cv2
import numpy as np
from keras.datasets import mnist



classifier = load_model('8_mnist_simple_cnn_10_Epochs.h5')

(x_train, y_train), (x_test, y_test)  = mnist.load_data()


def draw_test(name, pred, input_im):
    BLACK = [0,0,0]
    expanded_image = cv2.copyMakeBorder(input_im, 0, 0, 0, imageL.shape[0] ,cv2.BORDER_CONSTANT,value=BLACK)
    expanded_image = cv2.cvtColor(expanded_image, cv2.COLOR_GRAY2BGR)
    cv2.putText(expanded_image, str(pred), (152, 70) , cv2.FONT_HERSHEY_COMPLEX_SMALL,2, (0,255,0), 2)
    cv2.imshow(name, expanded_image)


input_im = cv2.imread('IMG_5926.JPG',0)
input_im = cv2.resize(input_im, (28, 28), interpolation=cv2.INTER_AREA)
input_im = cv2.bitwise_not(input_im)
for i in range(0,10):
    rand = np.random.randint(0,len(x_test))
    # input_im = x_test[rand]

    imageL = cv2.resize(input_im, None, fx=4, fy=4, interpolation = cv2.INTER_CUBIC)
    input_im = input_im.reshape(1,28,28,1) 
    
    ## Get Prediction
    # res = str(classifier.predict(input_im, 1, verbose = 0)[0])

    predict_x = classifier.predict(input_im, 1, verbose = 0)[0]

    res = str(np.argmax(predict_x,axis=-1))

    draw_test("Prediction", res, imageL) 
    cv2.waitKey(0)

cv2.destroyAllWindows()