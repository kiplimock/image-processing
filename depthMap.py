import cv2
import numpy as np
import matplotlib.pyplot as plt

def computeDepthMap(imgL, imgR):
    
    # ------------------------------------- #
    # SETUP
    # ------------------------------------- #
    imgL = cv2.GaussianBlur(imgL, (5,5), 0)
    imgR = cv2.GaussianBlur(imgR, (5,5), 0)

    # read camera data
    data = cv2.FileStorage('stereo_params.yml', cv2.FILE_STORAGE_READ)
    keys = ["K1", "K2", "D1", "D2", "R1", "R2", "P1", "P2", "T"]
    [K1, K2, D1, D2, R1, R2, P1, P2, T] = [data.getNode(key).mat() for key in keys]

    '''
    We know that

            |f  0   cx1  0|
    P1 =    |0  f   cy   0|
            |0  f   1    0|

    and 

            |f  0   cx2  Tx*f|
    P2 =    |0  f   cy   0   |
            |0  f   1    0   |

    and in our case, cx1 = cx2 = cx
    '''

    f = K1[0,0]
    Tx = T[0,0]
    P1 = np.hstack((K1, np.array([[0],[0],[0]])))
    P2 = np.hstack((K2, np.array([[Tx*f],[0],[0]])))


    # ------------------------------------- #
    # STEREO RECTIFICATION
    # ------------------------------------- #
    h1, w1 = imgL.shape

    # rectify images using initUndistortRectifyMap
    xmap1, ymap1 = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (w1,h1), cv2.CV_32FC1)
    xmap2, ymap2 = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (w1,h1), cv2.CV_32FC1)

    imgL_rectified = cv2.remap(imgL, xmap1, ymap1, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
    imgR_rectified = cv2.remap(imgR, xmap2, ymap2, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

    # -------------------------------- #
    # COMPUTE DISPARITY MAP
    # -------------------------------- #

    # Matched blocked size
    block_size = 11
    min_disp = -128
    max_disp = 128
    num_disp = max_disp - min_disp
    uniquenessRatio = 5
    speckleWindowSize = 200
    speckleRange = 2
    disp12MaxDiff = 0

    stereo = cv2.StereoSGBM_create(
                    minDisparity=min_disp,
                    numDisparities=num_disp,
                    blockSize=block_size,
                    uniquenessRatio=uniquenessRatio,
                    speckleWindowSize=speckleWindowSize,
                    speckleRange=speckleRange,
                    disp12MaxDiff=disp12MaxDiff,
                    P1=8 * block_size * block_size,
                    P2=32 * block_size * block_size
                    )

    disparity_SGBM = stereo.compute(imgL_rectified, imgR_rectified)
    disparity_SGBM = cv2.normalize(disparity_SGBM,
                            disparity_SGBM,
                            beta=0,
                            alpha=255,
                            norm_type=cv2.NORM_MINMAX)
    disparity_SGBM = np.uint8(disparity_SGBM)
    # cv2.imwrite('disparitySGBM.png', disparity_SGBM)


    # -------------------------------- #
    # FILTER DISPARITY MAP
    # -------------------------------- #
    right_matcher = cv2.ximgproc.createRightMatcher(stereo)
    # filter parameters
    lmbda = 80000
    sigma = 1.5
    visual_multiplier = 6

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    dispL = stereo.compute(imgL_rectified, imgR_rectified)
    dispR = right_matcher.compute(imgR_rectified, imgL_rectified)

    dispL = np.int16(dispL)
    dispR = np.int16(dispR)

    filteredImg = wls_filter.filter(dispL, imgL_rectified, None, dispR)
    # Normalize the values to the range 0-255 for a grayscale image
    filteredImg = cv2.normalize(filteredImg,
                            filteredImg,
                            beta=0,
                            alpha=255,
                            norm_type=cv2.NORM_MINMAX)
    filteredImg = np.uint8(filteredImg)
    # filteredImg[maskL_rectified == 0] = 0
    return filteredImg


def extractDisparityValues(left_ims, right_ims):

    # ------------------------------------- #
    # SETUP
    # ------------------------------------- #

    # read camera data
    data = cv2.FileStorage('stereo_params.yml', cv2.FILE_STORAGE_READ)
    keys = ["K1", "K2", "D1", "D2", "R1", "R2", "P1", "P2", "T"]
    [K1, K2, D1, D2, R1, R2, P1, P2, T] = [data.getNode(key).mat() for key in keys]

    '''
    We know that

            |f  0   cx1  0|
    P1 =    |0  f   cy   0|
            |0  f   1    0|

    and 

            |f  0   cx2  Tx*f|
    P2 =    |0  f   cy   0   |
            |0  f   1    0   |

    and in our case, cx1 = cx2 = cx
    '''

    f = K1[0,0]
    Tx = T[0,0]
    P1 = np.hstack((K1, np.array([[0],[0],[0]])))
    P2 = np.hstack((K2, np.array([[Tx*f],[0],[0]])))

    # ------------------------------------- #
    # STEREO RECTIFICATION
    # ------------------------------------- #
    disparities = []

    for file_L, file_R in zip(left_ims, right_ims):
            
            imgL = cv2.imread(file_L, 0)
            imgR = cv2.imread(file_R, 0)

            h1, w1 = imgL.shape

            # rectify images using initUndistortRectifyMap
            xmap1, ymap1 = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (w1,h1), cv2.CV_32FC1)
            xmap2, ymap2 = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (w1,h1), cv2.CV_32FC1)

            imgL_rectified = cv2.remap(imgL, xmap1, ymap1, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
            imgR_rectified = cv2.remap(imgR, xmap2, ymap2, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
            remap_combined = np.concatenate((imgL_rectified, imgR_rectified), axis=1)

            # -------------------------------- #
            # COMPUTE DISPARITY MAP
            # -------------------------------- #

            # Matched blocked size
            block_size = 11
            min_disp = -128
            max_disp = 128

            num_disp = max_disp - min_disp
            uniquenessRatio = 5
            speckleWindowSize = 200
            speckleRange = 2
            disp12MaxDiff = 0

            stereo = cv2.StereoSGBM_create(
                            minDisparity=min_disp,
                            numDisparities=num_disp,
                            blockSize=block_size,
                            uniquenessRatio=uniquenessRatio,
                            speckleWindowSize=speckleWindowSize,
                            speckleRange=speckleRange,
                            disp12MaxDiff=disp12MaxDiff,
                            P1=8 * block_size * block_size,
                            P2=32 * block_size * block_size
                            )

            disparity_SGBM = stereo.compute(imgL_rectified, imgR_rectified)
            disparity_SGBM = cv2.normalize(disparity_SGBM,
                                    disparity_SGBM,
                                    beta=0,
                                    alpha=255,
                                    norm_type=cv2.NORM_MINMAX)
            disparity_SGBM = np.uint8(disparity_SGBM)
            # cv2.imwrite('disparitySGBM.png', disparity_SGBM)


            # -------------------------------- #
            # FILTER DISPARITY MAP
            # -------------------------------- #
            right_matcher = cv2.ximgproc.createRightMatcher(stereo)
            # filter parameters
            lmbda = 80000
            sigma = 1.5
            visual_multiplier = 6

            wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
            wls_filter.setLambda(lmbda)
            wls_filter.setSigmaColor(sigma)

            dispL = stereo.compute(imgL_rectified, imgR_rectified)
            dispR = right_matcher.compute(imgR_rectified, imgL_rectified)

            dispL = np.int16(dispL)
            dispR = np.int16(dispR)

            filteredImg = wls_filter.filter(dispL, imgL_rectified, None, dispR)
            # Normalize the values to the range 0-255 for a grayscale image
            filteredImg = cv2.normalize(filteredImg,
                                    filteredImg,
                                    beta=0,
                                    alpha=255,
                                    norm_type=cv2.NORM_MINMAX)
            filteredImg = np.uint8(filteredImg)
            disparities.append(filteredImg[359, 748])
            return disparities




def maskDisparityMaps(left_ims, right_ims, left_masks):

        # ------------------------------------- #
        # SETUP
        # ------------------------------------- #

        # read camera data
        data = cv2.FileStorage('stereo_params_v7.yml', cv2.FILE_STORAGE_READ)
        keys = ["K1", "K2", "D1", "D2", "R1", "R2", "P1", "P2", "T"]
        [K1, K2, D1, D2, R1, R2, P1, P2, T] = [data.getNode(key).mat() for key in keys]

        '''
        We know that

                |f  0   cx1  0|
        P1 =    |0  f   cy   0|
                |0  f   1    0|

        and 

                |f  0   cx2  Tx*f|
        P2 =    |0  f   cy   0   |
                |0  f   1    0   |

        and in our case, cx1 = cx2 = cx
        '''

        f = K1[0,0]
        Tx = T[0,0]
        P1 = np.hstack((K1, np.array([[0],[0],[0]])))
        P2 = np.hstack((K2, np.array([[Tx*f],[0],[0]])))


        # ------------------------------------- #
        # STEREO RECTIFICATION
        # ------------------------------------- #
        i = 1
        for file_L, file_R, mask_file_L in zip(left_ims, right_ims, left_masks):

                imgL = cv2.imread(file_L, 0)
                imgR = cv2.imread(file_R, 0)
                maskL = cv2.imread(mask_file_L, 0)

                h1, w1 = imgL.shape
                h2, w2 = imgR.shape


                # rectify images using initUndistortRectifyMap
                xmap1, ymap1 = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (w1,h1), cv2.CV_32FC1)
                xmap2, ymap2 = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (w1,h1), cv2.CV_32FC1)

                imgL_rectified = cv2.remap(imgL, xmap1, ymap1, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
                maskL_rectified = cv2.remap(maskL, xmap1, ymap1, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
                imgR_rectified = cv2.remap(imgR, xmap2, ymap2, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
                remap_combined = np.hstack((imgL_rectified, imgR_rectified))

                # -------------------------------- #
                # COMPUTE DISPARITY MAP
                # -------------------------------- #

                # Matched blocked size
                block_size = 11
                min_disp = -128
                max_disp = 128

                num_disp = max_disp - min_disp
                uniquenessRatio = 5
                speckleWindowSize = 200
                speckleRange = 2
                disp12MaxDiff = 0

                stereo = cv2.StereoSGBM_create(
                minDisparity=min_disp,
                numDisparities=num_disp,
                blockSize=block_size,
                uniquenessRatio=uniquenessRatio,
                speckleWindowSize=speckleWindowSize,
                speckleRange=speckleRange,
                disp12MaxDiff=disp12MaxDiff,
                P1=8 * block_size * block_size,
                P2=32 * block_size * block_size
                )

                disparity_SGBM = stereo.compute(imgL_rectified, imgR_rectified)
                disparity_SGBM = cv2.normalize(disparity_SGBM,
                                        disparity_SGBM,
                                        beta=0,
                                        alpha=255,
                                        norm_type=cv2.NORM_MINMAX)
                disparity_SGBM = np.uint8(disparity_SGBM)
                # cv2.imwrite('disparitySGBM.png', disparity_SGBM)


                # -------------------------------- #
                # FILTER DISPARITY MAP
                # -------------------------------- #
                right_matcher = cv2.ximgproc.createRightMatcher(stereo)
                # filter parameters
                lmbda = 8000
                sigma = 1.5
                visual_multiplier = 6

                wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
                wls_filter.setLambda(lmbda)
                wls_filter.setSigmaColor(sigma)

                dispL = stereo.compute(imgL_rectified, imgR_rectified)
                dispR = right_matcher.compute(imgR_rectified, imgL_rectified)

                dispL = np.int16(dispL)
                dispR = np.int16(dispR)

                filteredImg = wls_filter.filter(dispL, imgL_rectified, None, dispR)

                # Normalize the values to the range 0-255 for a grayscale image
                filteredImg = cv2.normalize(filteredImg,
                                        filteredImg,
                                        beta=0,
                                        alpha=255,
                                        norm_type=cv2.NORM_MINMAX)
                filteredImg = np.uint8(filteredImg)
                # cv2.imwrite('filteredSGBM.png', filteredImg)

                # save disparity maps
                cv2.imwrite('trees/filtered_disparity_maps/filtered_disparity_' + str(i) + '.png', filteredImg)
                filteredImg[maskL_rectified == 0] = 0
                cv2.imwrite('trees/masked_disparity_maps/masked_disparity_' + str(i) + '.png', filteredImg)
                i += 1
