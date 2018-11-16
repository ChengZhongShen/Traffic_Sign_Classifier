def resize_images():
    import cv2
    import matplotlib.pyplot as plt 

    import sys, os

    path = './examples_2/traffic_sign_from_web_cropped/'
    files = os.listdir(path)

    for file in files:
        img = cv2.imread(path+file)
        # plt.imshow(img)
        # plt.show()
        # sys.exit()
        # resized_img = cv2.resize(img, (32,32))
        # cv2.imwrite('./examples_2/'+file, resized_img)
        resized_img = cv2.resize(img, (200,200))
        cv2.imwrite('./examples/'+file, resized_img)

def output_images():
    import sys, os
    import numpy as np 
    import matplotlib.pyplot as plt 
    from matplotlib.gridspec import GridSpec
    import pickle

    images_from_web = np.zeros((12,32,32,3), dtype=np.uint8)
    y_images_from_web = np.zeros(12, dtype=np.uint8)

    path = './examples_2/'
    files = os.listdir(path)

    for index, filename in enumerate(files):
        if filename[-3:] != 'jpg': # skip non jpg filename
            continue
        img = plt.imread(path+filename)
        images_from_web[index] = img
        y_images_from_web[index]= int(filename.split('_')[0])

    samples_fig = plt.figure(figsize=(16,12))
    rows = 3
    cols = 4
    count = 0
    gs = GridSpec(rows,cols)
    for i in range(rows):
        for j in range(cols):
            ax = samples_fig.add_subplot(gs[i,j])
            ax.imshow(images_from_web[count])
            ax.set_xticks([])
            ax.set_yticks([])
            count += 1
    plt.show()

    print(y_images_from_web)

    images_web = [images_from_web, y_images_from_web]
    with open('images_web.p', 'wb') as f:
        pickle.dump(images_web, f)

resize_images()
# output_images()

import matplotlib.pyplot as plt
plt.imshow()