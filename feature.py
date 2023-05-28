import cv2
import matplotlib.pyplot as plt
import copy
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from skimage import morphology

def feature(p_path, method='sift', show=True,x_start=0,y_start=0,x_end=500,y_end=500,skeleton=False):
    img = cv2.imread(p_path,cv2.IMREAD_GRAYSCALE)

    ret,img=cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    if skeleton:
        img=255-img

        #实施骨架算法
        skeleton =morphology.skeletonize(img)

        skeleton=255-skeleton
        img=255-img
        skeleton[skeleton!=255]=0
        img=skeleton.astype(np.uint8)

    img = img[x_start:x_end,y_start:y_end]
    if method == 'sift':
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(img, None)
        kp_image1 = cv2.drawKeypoints(img, kp, None)
        if show:
            plt.figure()
            plt.imshow(kp_image1)
            plt.show()
        return sift.detectAndCompute(img, None)

    if method == 'surf':
        surf = cv2.xfeatures2d.SURF_create(4000)
        kp, des = surf.detectAndCompute(img, None)
        img1 = cv2.drawKeypoints(img, kp, None, (255, 0, 0), 4)
        if show:
            plt.imshow(img1)
            plt.show()
        return surf.detectAndCompute(img, None)

    if method == 'orb':
        orb = cv2.ORB_create()
        kp, des = orb.detectAndCompute(img, None)
        return orb.detectAndCompute(img, None)

    if method == 'shi-tomasi':
        corners = cv2.goodFeaturesToTrack(img, 30, 0.3, 5)  # 返回的结果是 [[ a., b.]] 两层括号的数组。
        if show:
            for i in corners:
                x, y = i.ravel()
                x = int(x)
                y = int(y)
                cv2.circle(img, (x, y), 5, 170, -1)  # 在角点处画圆，半径为2，红色，线宽默认，利于显示
            plt.imshow(img)
            plt.show()
        return corners

    if method == 'hog':
        block_size = (16, 16)  # 每个块的大小
        block_stride = (8, 8)  # 每次移动的距离
        cell_size = (8, 8)  # 每个块里的小格子的大小
        bins = 9
        x_cells = img.shape[1] // cell_size[0]
        y_cells = img.shape[0] // cell_size[1]
        n1 = int(block_size[0] / cell_size[0])
        n2 = int(block_size[1] / cell_size[1])
        win_size = (x_cells * cell_size[0], y_cells * cell_size[1])
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, bins)
        # hog = cv2.HOGDescriptor()
        # win_stride = (8, 8)
        # padding = (8, 8)
        # test_hog = hog.compute(img, win_stride, padding)
        test_hog = hog.compute(img)
        tot_bx = int(x_cells - n1 + 1)
        tot_by = int(y_cells - n2 + 1)
        hog_descriptor_reshaped = test_hog.reshape(tot_bx, tot_by, n1, n2, bins).transpose((1, 0, 2, 3, 4))

        if show:

            ave_grad = np.zeros((y_cells, x_cells, bins))
            hist_counter = np.zeros((y_cells, x_cells, 1))
            for i in range(n1):
                for j in range(n2):
                    ave_grad[i:tot_by + i, j:tot_bx + j] += hog_descriptor_reshaped[:, :, i, j, :]

                    hist_counter[i:tot_by + i, j:tot_bx + j] += 1
            ave_grad /= hist_counter
            len_vecs = ave_grad.shape[0] * ave_grad.shape[1] * ave_grad.shape[2]
            deg = np.linspace(0, np.pi, bins, endpoint=False)
            U = np.zeros(len_vecs)
            V = np.zeros(len_vecs)
            X = np.zeros(len_vecs)
            Y = np.zeros(len_vecs)
            counter = 0

            for i in range(ave_grad.shape[0]):
                for j in range(ave_grad.shape[1]):
                    for k in range(ave_grad.shape[2]):
                        U[counter] = ave_grad[i, j, k] * np.cos(deg[k])
                        V[counter] = ave_grad[i, j, k] * np.sin(deg[k])

                        X[counter] = (cell_size[0] / 2) + (cell_size[0] * i)
                        Y[counter] = (cell_size[1] / 2) + (cell_size[1] * j)

                        counter = counter + 1
            angle_axis = np.linspace(0, 180, bins, endpoint=False)
            angle_axis += ((angle_axis[1] - angle_axis[0]) / 2)
            fig, ((a, b), (c, d)) = plt.subplots(2, 2)

            # Set the title of each subplot
            a.set(title='Gray Scale Image\n(Click to Zoom)')
            b.set(title='HOG Descriptor\n(Click to Zoom)')
            c.set(title='Zoom Window', xlim=(0, 18), ylim=(0, 18), autoscale_on=False)
            d.set(title='Histogram of Gradients')

            a.imshow(img, cmap='gray')
            a.set_aspect(aspect=1)

            b.quiver(Y, X, U, V, color='white', headwidth=0, headlength=0, scale_units='inches', scale=5)
            b.invert_yaxis()
            b.set_aspect(aspect=1)
            b.set_facecolor('black')

            def onpress(event):

                # Unless the left mouse button is pressed do nothing
                if event.button != 1:
                    return

                # Only accept clicks for subplots a and b
                if event.inaxes in [a, b]:
                    # Get mouse click coordinates
                    x, y = event.xdata, event.ydata

                    # Select the cell closest to the mouse click coordinates
                    cell_num_x = np.uint32(x / cell_size[0])
                    cell_num_y = np.uint32(y / cell_size[1])

                    # Set the edge coordinates of the rectangle patch
                    edgex = x - (x % cell_size[0])
                    edgey = y - (y % cell_size[1])

                    # Create a rectangle patch that matches the cell selected above
                    rect = patches.Rectangle((edgex, edgey),
                                            cell_size[0], cell_size[1],
                                            linewidth=1,
                                            edgecolor='magenta',
                                            facecolor='none')

                    # A single patch can only be used in a single plot. Create copies
                    # of the patch to use in the other subplots
                    rect2 = copy.copy(rect)
                    rect3 = copy.copy(rect)

                    # Update all subplots
                    a.clear()
                    a.set(title='Gray Scale Image\n(Click to Zoom)')
                    a.imshow(img, cmap='gray')
                    a.set_aspect(aspect=1)
                    a.add_patch(rect)

                    b.clear()
                    b.set(title='HOG Descriptor\n(Click to Zoom)')
                    b.quiver(Y, X, U, V, color='white', headwidth=0, headlength=0, scale_units='inches', scale=5)
                    b.invert_yaxis()
                    b.set_aspect(aspect=1)
                    b.set_facecolor('black')
                    b.add_patch(rect2)

                    c.clear()
                    c.set(title='Zoom Window')
                    c.quiver(Y, X, U, V, color='white', headwidth=0, headlength=0, scale_units='inches', scale=1)
                    c.set_xlim(edgex - cell_size[0], edgex + (2 * cell_size[0]))
                    c.set_ylim(edgey - cell_size[1], edgey + (2 * cell_size[1]))
                    c.invert_yaxis()
                    c.set_aspect(aspect=1)
                    c.set_facecolor('black')
                    c.add_patch(rect3)

                    d.clear()
                    d.set(title='Histogram of Gradients')
                    d.grid()
                    d.set_xlim(0, 180)
                    d.set_xticks(angle_axis)
                    d.set_xlabel('Angle')
                    d.bar(angle_axis,
                        ave_grad[cell_num_y, cell_num_x, :],
                        180 // bins,
                        align='center',
                        alpha=0.5,
                        linewidth=1.2,
                        edgecolor='k')

                    fig.canvas.draw()

            # Create a connection between the figure and the mouse click
            fig.canvas.mpl_connect('button_press_event', onpress)
            
            plt.show()

        return hog_descriptor_reshaped


def match(path1, path2, method='sift', p=2, r=0.85, show=True,skeletoned=False):
    good_match = []
    img1 = cv2.imread(path1,cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(path2,cv2.IMREAD_GRAYSCALE)


    ret,img1=cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY)
    ret,img2=cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)
    if skeletoned:
        img1=255-img1

        #实施骨架算法
        skeleton =morphology.skeletonize(img1)

        skeleton=255-skeleton
        img1=255-img1
        skeleton[skeleton!=255]=0
        img1=skeleton.astype(np.uint8)
        img2=255-img2

        #实施骨架算法
        skeleton =morphology.skeletonize(img2)

        skeleton=255-skeleton
        img2=255-img2
        skeleton[skeleton!=255]=0
        img2=skeleton.astype(np.uint8)

    img1=img1.astype(np.uint8)
    img2=img2.astype(np.uint8)
    kp1, des1 = feature(path1, method, show,skeleton=skeletoned)
    kp2, des2 = feature(path2, method, show,skeleton=skeletoned)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=p)
    for i in matches:
        if len(i)<2:
            return 0
    for m1, m2 in matches:
        if m1.distance < r * m2.distance:
            good_match.append([m1])
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_match, None, flags=2)
    if show:
        plt.imshow(img3)
        plt.show()
    return len(good_match) / len(matches)
