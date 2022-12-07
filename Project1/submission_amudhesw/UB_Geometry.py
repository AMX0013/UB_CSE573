import numpy as np
from typing import List, Tuple
import cv2

from collections import namedtuple

from cv2 import cvtColor, COLOR_BGR2GRAY, TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER, \
    findChessboardCorners, cornerSubPix, drawChessboardCorners

'''
Please do Not change or add any imports. 
'''

#task1

def findRot_xyz2XYZ(alpha: float, beta: float, gamma: float) -> np.ndarray:
    '''
    Args:
        alpha, beta, gamma: They are the rotation angles along x, y and z axis respectly.
            Note that they are angles, not radians.
    Return:
        A 3x3 numpy array represents the rotation matrix from xyz to XYZ.

    '''
    rot_xyz2XYZ = np.eye(3).astype(float)

    # Your implementation
    zeroMat = [ [0,0,0],[0,0,0], [0,0,0]]  #to deal with -0 showing up in place of 0 due to np.sin(0) or np.cos(pi/2)

    cosA = np.cos(np.deg2rad(alpha)) #around z axis
    sinA = np.sin(np.deg2rad(alpha))

    cosB = np.cos(np.deg2rad(beta)) #around x axis
    sinB = np.sin(np.deg2rad(beta))

    cosG = np.cos(np.deg2rad(gamma)) #around Z axis
    sinG = np.sin(np.deg2rad(gamma))


    RotateOnZbyAlpha =[ [ cosA , -sinA  , 0  ] , [ sinA , cosA , 0 ] , [ 0 , 0  , 1  ] ] #1

    RotateOnXbyBeta = [ [ 1  , 0  ,  0 ] , [ 0 , cosB  , -sinB  ] , [ 0 , sinB  , cosB  ] ] #2

    RotateOnZbyGamma =[ [ cosG , -sinG  , 0  ] , [ sinG , cosG , 0 ] , [ 0 , 0  , 1  ] ]  #3


    rot_xyz2XYZ = np.matmul(rot_xyz2XYZ,RotateOnZbyAlpha)

    rot_xyz2XYZ = np.matmul(rot_xyz2XYZ,RotateOnXbyBeta)

    rot_xyz2XYZ = np.matmul(rot_xyz2XYZ,RotateOnZbyGamma)

    #return np.around(rot_xyz2XYZ , decimals= 5) 
    
    return np.add(zeroMat, np.around(rot_xyz2XYZ , decimals= 5) )


def findRot_XYZ2xyz(alpha: float, beta: float, gamma: float) -> np.ndarray:
    '''
    Args:
        alpha, beta, gamma: They are the rotation angles of the 3 step respectly.
            Note that they are angles, not radians.
    Return:
        A 3x3 numpy array represents the rotation matrix from XYZ to xyz.

    '''
    rot_XYZ2xyz = np.eye(3).astype(float)
    
    # Your implementation
    zeroMat = [ [0,0,0],[0,0,0], [0,0,0]]  #to deal with -0 showing up in place of 0 due to np.sin(0) or np.cos(pi/2)
    alpha = -alpha
    beta = -beta
    gamma = -gamma

    cosA = np.cos(np.deg2rad(alpha)) #around z axis
    sinA = np.sin(np.deg2rad(alpha))

    cosB = np.cos(np.deg2rad(beta)) #around x axis
    sinB = np.sin(np.deg2rad(beta))

    cosG = np.cos(np.deg2rad(gamma)) #around Z axis
    sinG = np.sin(np.deg2rad(gamma))


    RotateOnZbyAlpha =[ [ cosA , -sinA  , 0  ] , [ sinA , cosA , 0 ] , [ 0 , 0  , 1  ] ] #1

    RotateOnXbyBeta = [ [ 1  , 0  ,  0 ] , [ 0 , cosB  , -sinB  ] , [ 0 , sinB  , cosB  ] ] #2

    RotateOnZbyGamma =[ [ cosG , -sinG  , 0  ] , [ sinG , cosG , 0 ] , [ 0 , 0  , 1  ] ]  #3

    rot_XYZ2xyz = np.matmul(rot_XYZ2xyz,RotateOnZbyGamma) #RotateOnZbyGamma
    rot_XYZ2xyz = np.matmul(rot_XYZ2xyz,RotateOnXbyBeta)
    rot_XYZ2xyz = np.matmul(rot_XYZ2xyz,RotateOnZbyAlpha) #RotateOnZbyAlpha


    #return np.around(rot_XYZ2xyz , decimals= 5)   
    
    return np.add(zeroMat, np.around(rot_XYZ2xyz , decimals= 5) )

"""
If your implementation requires implementing other functions. Please implement all the functions you design under here.
But remember the above "findRot_xyz2XYZ()" and "findRot_XYZ2xyz()" functions are the only 2 function that will be called in task1.py.
"""

# Your functions for task1






#--------------------------------------------------------------------------------------------------------------
# task2:

def find_corner_img_coord(image: np.ndarray) -> np.ndarray:
    '''
    Args: 
        image: Input image of size MxNx3. M is the height of the image. N is the width of the image. 3 is the channel of the image.
    Return:
        A numpy array of size 32x2 that represents the 32 checkerboard corners' pixel coordinates. 
        The pixel coordinate is defined such that the of top-left corner is (0, 0) and the bottom-right corner of the image is (N, M). 
    '''
    img_coord = np.zeros([32, 2], dtype=float)

    # Your implementation
    colorImg = image
    image = cvtColor(image, COLOR_BGR2GRAY )
    #print(image.shape)
    
    
    terminationCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)


    success , corners = cv2.findChessboardCorners(image= image,patternSize=(4,9))
    cornerList=corners
    if success:
        
        cornerList = np.delete(corners,np.s_[16:20],0)
        cornerList.flatten

        #print(cornerList)

        cv2.cornerSubPix(image= image, corners= cornerList, winSize=(4,8),zeroZone=(0,0), criteria=terminationCriteria  )  

        cv2.drawChessboardCorners(image=colorImg, patternSize= (4,8) , corners= cornerList, patternWasFound=success)

        #to display image with corners
        # cv2.imshow(winname='bgr2gray', mat=colorImg )
        # cv2.waitKey(0)
        
        # cornerList.ravel()

        img_coord = np.array([corner for [corner] in cornerList]).reshape(32,2)
        #print(corners)
        return img_coord


def find_corner_world_coord(img_coord: np.ndarray) -> np.ndarray:
    '''
    You can output the world coord manually or through some algorithms you design. Your output should be the same order with img_coord.
    Args: 
        img_coord: The image coordinate of the corners. Note that you do not required to use this as input, 
        as long as your output is in the same order with img_coord.
    Return:
        A numpy array of size 32x3 that represents the 32 checkerboard corners' pixel coordinates. 
        The world coordinate or each point should be in form of (x, y, z). 
        The axis of the world coordinate system are given in the image. The output results should be in milimeters.
    '''
    world_coord = np.zeros([32, 3], dtype=float)
    
    # Your implementation
    
    wcResult = worldCoordinatePopulator()
    world_coord= np.array(wcResult,np.float32)
    #print(world_coord,"wc")
    return world_coord


def find_intrinsic(img_coord: np.ndarray, world_coord: np.ndarray) -> Tuple[float, float, float, float]:
    '''
    Use the image coordinates and world coordinates of the 32 point to calculate the intrinsic parameters.
    Args: 
        img_coord: The image coordinate of the 32 corners. This is a 32x2 numpy array.
        world_coord: The world coordinate of the 32 corners. This is a 32x3 numpy array.
    Returns:
        fx, fy: Focal length. 
        (cx, cy): Principal point of the camera (in pixel coordinate).
    '''

    fx: float = 0
    fy: float = 0
    cx: float = 0
    cy: float = 0

    # Your implementation

    fx,fy,cx,cy, _ ,_ = getCameraCalibration(img_coord, world_coord)

    return fx, fy, cx, cy


def find_extrinsic(img_coord: np.ndarray, world_coord: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Use the image coordinates, world coordinates of the 32 point and the intrinsic parameters to calculate the extrinsic parameters.
    Args: 
        img_coord: The image coordinate of the 32 corners. This is a 32x2 numpy array.
        world_coord: The world coordinate of the 32 corners. This is a 32x3 numpy array.
    Returns:
        R: The rotation matrix of the extrinsic parameters. It is a 3x3 numpy array.
        T: The translation matrix of the extrinsic parameters. It is a 1-dimensional numpy array with length of 3.
    '''

    R = np.eye(3).astype(float)
    T = np.zeros(3, dtype=float)

    # Your implementation
    _ , _ , _ , _ , R,T = getCameraCalibration(img_coord, world_coord)


    return R, T


"""
If your implementation requires implementing other functions. Please implement all the functions you design under here.
But remember the above 4 functions are the only ones that will be called in task2.py.
"""

# Your functions for task2
def worldCoordinatePopulator():
    #print("flowCheckmethod")
    wc = []
    for i in range(2):
        #print(i)
        if i == 0:
            #Point on XZ plane : Y=0
            Y=0
            X=10
            for XY in range(4,0,-1):
                for Z in range(40,0,-10):
                    #print(X*XY,Y*XY,Z)
                    wc.append([X*XY,Y*XY,Z])

        else:
            #Point on YZ plane : X=0
            Y=10
            X=0
            for XY in range(1,5):
                for Z in range(40,0,-10):
                    #print(X*XY,Y*XY,Z)
                    wc.append([X*XY,Y*XY,Z])
    
    return wc

def getCameraCalibration(img_coord: np.ndarray, world_coord: np.ndarray):
    
     
    matrix64by12 = np.zeros( (len(world_coord)*2,12 ) )
    rowCount = 0

    for i in range(len(world_coord)): #32 world coordinates,32 img coordinates, 2 entries per iteration
        row = np.matrix([world_coord[i][0], world_coord[i][1], world_coord[i][2],1, 0.0, 0.0, 0.0, 0.0, \
        -img_coord[i][0] * world_coord[i][0], -img_coord[i][0] * world_coord[i][1], -img_coord[i][0] * world_coord[i][2], -img_coord[i][0]])

        matrix64by12[rowCount] = row
        rowCount +=1

        row = np.matrix([0.0, 0.0, 0.0, 0.0, world_coord[i][0], world_coord[i][1], world_coord[i][2], 1, \
        -img_coord[i][1] * world_coord[i][0], -img_coord[i][1] * world_coord[i][1], -img_coord[i][1] * world_coord[i][2], -img_coord[i][1]])

        matrix64by12[rowCount] = row
        rowCount +=1

    u , s , vh = np.linalg.svd(matrix64by12)
    m = vh[-1].reshape(3,4)
    print(m)
    m1 = m[0]
    m2 = m[1]
    m3 = m[2]


    ox = np.matmul(m1.T , m3) 
    oy = np.matmul(m2.T , m3)

    # print("ox = \n",ox)
    # print("oy = \n",oy )
    fx = np.sqrt( np.matmul(m1.T , m1)  -(ox*ox) )
    fy = np.sqrt( np.matmul(m2.T , m2)  -(oy*oy) )

    # print("fx = \n",fx)
    # print("fy = \n",fy)    

    mIntrinsic = np.matrix( [ [fx, 0.0, ox], [ 0.0, fy,  oy], [0, 0, 1] ]  )
    mExtrinsic = np.matmul( mIntrinsic.I , m )
    mExtrinsic_Rotation = mExtrinsic[:,:-1]
    mExtrinsic_Translation =  mExtrinsic[:,-1]
    #print("mExtrinsic_Rotation > \n",mExtrinsic_Rotation)
    #print("mExtrinsic_Translation > \n",mExtrinsic_Translation)



 
    return (fx,fy,ox,oy,mExtrinsic_Rotation,mExtrinsic_Translation)
    
    
    
    











        







#---------------------------------------------------------------------------------------------------------------------