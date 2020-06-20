if __name__ == '__main__':
        print('my_homography')
        #downSampleRate = 4
        image1 = cv2.imread('data/incline_L.png')
        image2 = cv2.imread('data/incline_R.png')
        im1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        im2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        #im1 = im1[::downSampleRate, ::downSampleRate, :]  # downSample because the memory problems
        #im2 = im2[::downSampleRate, ::downSampleRate, :]  # downSample because the memory problems

        H2to1 = np.array([[1.69424090e-03,  1.92939042e-05,  9.98620205e-01],
                   [-2.63154587e-04,  2.48851693e-03, -5.23534795e-02],
                   [-1.14541269e-06,  1.20477243e-07,  2.76877412e-03]])

        # part 2.3
        H_trans, out_size, axis_arr = Translation(im1, H2to1)  # not in HW

        warp_im1 = warpH(im1, H_trans, out_size)
        plt.figure(2)
        plt.imshow(warp_im1)
        plt.show()

        warp_im1_scaled,im2_scaled = getScaled(im2, warp_im1, axis_arr)
        plt.imshow(warp_im1_scaled)
        plt.imshow(im2_scaled)
        panoramaTest = imageStitching(im2_scaled, warp_im1_scaled)
        plt.figure(3)
        plt.imshow(panoramaTest)
        plt.show()