import cv2
import os
import random


IMAGE_WIDTH = 320
IMAGE_HEIGHT = 480
CUT_IMAGE_NUM = 1
FACTOR = 1/2
# FACTOR = 1


class DataProcesser(object):
    def __init__(self):
        pass


    def cut_for_train(self, from_path, dest_path):
        if (not os.path.exists(from_path)):
            return None

        if(not os.path.exists(dest_path)):
            try:
                os.mkdir(dest_path)
            except:
                return None

        i = 0
        file_list = os.listdir(from_path)
        for file in file_list:
            full_path = os.path.join(from_path, file)
            image = cv2.imread(full_path)
            if(image is None):
                continue

            img_width, img_height, img_channel = image.shape
            if(img_width > IMAGE_WIDTH and img_height > IMAGE_HEIGHT):
                for j in range(CUT_IMAGE_NUM):
                    m = random.randint(0, img_width-IMAGE_WIDTH)
                    n = random.randint(0, img_height-IMAGE_HEIGHT)
                    new_image = image[m:m+IMAGE_WIDTH, n:n+IMAGE_HEIGHT]
                    # all save to jpg format
                    cv2.imwrite(os.path.join(dest_path,str(i) + '_' + str(j) + '.jpg'), new_image)

                i += 1


    def down_sampling_for_train(self, from_path, dest_path):
        if (not os.path.exists(from_path)):
            return None

        if (not os.path.exists(dest_path)):
            try:
                os.mkdir(dest_path)
            except:
                return None

        file_list = os.listdir(from_path)

        for file in file_list:
            full_path = os.path.join(from_path, file)
            image = cv2.imread(full_path)
            width, height, channel = image.shape
            # kernel_size = (5, 5);
            # sigma = 1.5;
            #kernel_size = (7, 7);
            #sigma = 3;
            # kernel_size = (17, 17);
            # sigma = 9;
            # image=cv2.GaussianBlur(image, kernel_size, sigma);
            # cv2.blur()


            # image = cv2.resize(image, (int(IMAGE_WIDTH*FACTOR),int(IMAGE_HEIGHT*FACTOR)), interpolation=cv2.INTER_CUBIC)
            image = cv2.resize(image, (int(height * FACTOR), int(width * FACTOR)), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(dest_path, file), image)



    def down_sampling(self, from_path, dest_path):
        if (not os.path.exists(from_path)):
            return None

        if (not os.path.exists(dest_path)):
            try:
                os.mkdir(dest_path)
            except:
                return None

        file_list = os.listdir(from_path)
        for file in file_list:
            full_path = os.path.join(from_path, file)
            image = cv2.imread(full_path)
            width, height, channel = image.shape
            image = cv2.resize(image, (int(height*FACTOR),int(width*FACTOR)), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(dest_path, file), image)


    # def up_sampling(self, from_path, dest_path):
    #     if (not os.path.exists(from_path)):
    #         return None
    #
    #     if (not os.path.exists(dest_path)):
    #         try:
    #             os.mkdir(dest_path)
    #         except:
    #             return None


        file_list = os.listdir(from_path)
        for file in file_list:
            full_path = os.path.join(from_path, file)
            image = cv2.imread(full_path)
            width, height, channel = image.shape
            image = cv2.resize(image, (int(height/FACTOR),int(width/FACTOR)), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(dest_path, file), image)



if __name__ == '__main__':
    data_processer = DataProcesser()
    data_processer.cut_for_train('/home/data1/11server/2K/Train/HR/', './imgs/hr')
    # data_processer.cut_for_train('./image/train/origin1', './image/train/hr')
    # data_processer.down_sampling_for_train('./image/train/cut', './image/train/down_sampling')
    data_processer.down_sampling_for_train('./imgs/hr', './imgs/lr')
    #data_processer.down_sampling('./image/test/origin', './image/test/down_sampling')
    #data_processer.up_sampling('./image/test/down_sampling', './image/test/up_sampling')