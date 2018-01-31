'''
for the transform  from viewpoint1 to abitrary viewpoint
viewpoint 8-12  orthoganal viewpoint x5
viewpoint 0-7 perspective viewpoint x8
'''
from __future__ import print_function,division

import numpy as np
import math

axis_X=np.array([1.0,0.0,0.0])
axis_Y=np.array([0.0,1.0,0.0])
axis_Z=np.array([0.0,0.0,1.0])

def Angle2Rad(angle):
    return math.pi*angle/180

def normalize(v):
    dim=v.size
    len2=0
    for coord in v:
        len2+=coord*coord
    return v/math.sqrt(len2)

def Translate(transform, v ):
    if(transform.shape!=(4,4) or v.size!=3):
        print ("Translate shape error!")
        return 0
    result=transform
    result[3]=transform[0] * v[0] + transform[1] * v[1] + transform[2] * v[2] + transform[3]
    return result


def Rotate(transform, v ,angle):
    #this anlgle should change to radian
    if(transform.shape!=(4,4) or v.size!=3):
        print ("Rotate shape error!")
        return 0
    cos = math.cos(Angle2Rad(angle))
    sin = math.sin(Angle2Rad(angle))
    axis= normalize(v)
    temp=axis*(1-cos)

    rotate=np.identity(4,dtype=np.float32)
    rotate[0][0] = cos + temp[0] * axis[0]
    rotate[0][1] = 0 + temp[0] * axis[1] + sin * axis[2]
    rotate[0][2] = 0 + temp[0] * axis[2] - sin * axis[1]

    rotate[1][0] = 0 + temp[1] * axis[0] - sin * axis[2]
    rotate[1][1] = cos + temp[1] * axis[1]
    rotate[1][2] = 0 + temp[1] * axis[2] + sin * axis[0]

    rotate[2][0] = 0 + temp[2] * axis[0] + sin * axis[1]
    rotate[2][1] = 0 + temp[2] * axis[1] - sin * axis[0]
    rotate[2][2] = cos + temp[2] * axis[2]

    Result=np.identity(4,dtype=np.float32)
    Result[0] = transform[0] * rotate[0][0] + transform[1] * rotate[0][1] + transform[2] * rotate[0][2]
    Result[1] = transform[0] * rotate[1][0] + transform[1] * rotate[1][1] + transform[2] * rotate[1][2]
    Result[2] = transform[0] * rotate[2][0] + transform[1] * rotate[2][1] + transform[2] * rotate[2][2]
    Result[3] = transform[3]
    return Result

def Scale(transform, v ):
    if(transform.shape!=(4,4) or v.size!=3):
        print ("Scale shape error!")
        return 0
    Result = np.identity(4, dtype=np.float32)
    Result[0] = transform[0] * v[0]
    Result[1] = transform[1] * v[1]
    Result[2] = transform[2] * v[2]
    Result[3] = transform[3]
    return Result

def LookAt(eye,center,up):
    if(eye.size!=3 or center.size!=3 or up.size!=3):
        print ("LookAt shape error")
        return 0
    f=normalize(eye-center)
    s=normalize(np.cross(up,f))
    u=normalize(np.cross(f,s))

    # result[col][row] when result[0] is the first column
    Result = np.identity(4, dtype=np.float32)
    Result[0][0] = s[0] #first row
    Result[1][0] = s[1]
    Result[2][0] = s[2]
    Result[0][1] = u[0] #second row
    Result[1][1] = u[1]
    Result[2][1] = u[2]
    Result[0][2] = f[0] #third row
    Result[1][2] = f[1]
    Result[2][2] = f[2]

    translate = np.identity(4, dtype=np.float32)
    translate[3][0] = -eye[0]
    translate[3][1] = -eye[1]
    translate[3][2] = -eye[2]
    Result=translate.dot(Result)

    return Result

def Perspective(fov,ar,zNear,zFar):
    print (ar)
    Result = np.zeros((4,4), dtype=np.float32)
    tanHalfFovy=math.tan(Angle2Rad(fov/2))

    Result[0][0] = 1.0 / (ar * tanHalfFovy)
    Result[1][1] = 1.0 / (tanHalfFovy)
    Result[2][2] = - (zFar + zNear) / (zFar - zNear)
    Result[2][3] = - 1.0
    Result[3][2] = - (2.0 * zFar * zNear) / (zFar - zNear)

    return Result

###########################################################
## uesd for binvox
def Rotate_orginal(transform,v,angle):
    result=Rotate(transform,v,angle)
    result=result[:3,:3].T
    return result

def Scale_orginal(transform,v):
    result=Scale(transform,v)
    result = result[:3, :3].T
    return result

## the transform matrix transform=[3x3]   used:transform.dot(coord)
#### return transform matrix[3x3] of ortho camera (reference camera 9 )
def trans9_8(): ##nice
    transform=np.identity(4)
    transform = Rotate(transform, axis_X, 90.0)
    transform = Rotate(transform, axis_Y, -90.0)  ##first rotate y -90 then rotate x +90
    transform=transform[:3,:3]
    return  transform.T

def trans8_9(): ##nice
    transform=np.identity(4)
    transform=Rotate(transform,axis_Y,90.0)
    transform = Rotate(transform, axis_X, -90.0)
    transform=transform[:3,:3]
    return  transform.T

def trans9_10():
    transform = np.identity(4)
    transform=Rotate(transform,axis_Y,-90)
    transform = transform[:3, :3]
    return transform.T

def trans10_9():
    transform = np.identity(4)
    transform=Rotate(transform,axis_Y,90)
    transform = transform[:3, :3]
    return transform.T

def trans9_11():
    transform = np.identity(4)
    transform=Rotate(transform,axis_Y,180)
    transform = transform[:3, :3]
    return transform.T

def trans11_9():
    transform = np.identity(4)
    transform=Rotate(transform,axis_Y,180)
    transform = transform[:3, :3]
    return transform.T

def trans9_12():
    transform = np.identity(4)
    transform=Rotate(transform,axis_Y,90)
    transform = transform[:3, :3]
    return transform.T

def trans12_9():
    transform = np.identity(4)
    transform=Rotate(transform,axis_Y,90)
    transform = transform[:3, :3]
    return transform.T

#### return transform matrix[3x3] of prespetive camera
def trans9_0():
    transform = np.identity(4)
    transform=Rotate(transform,axis_X,45)
    transform = Rotate(transform, axis_Y, -51.34) ##first rotate y -51.34 then rotate x 45
    transform = transform[:3, :3]
    return transform.T

def trans0_9():
    transform = np.identity(4)
    transform = Rotate(transform, axis_Y,51.34)
    transform = Rotate(transform, axis_X, -45)
    transform = transform[:3, :3]
    return transform.T

def trans9_1():
    transform = np.identity(4)
    transform=Rotate(transform,axis_X,45)
    transform = Rotate(transform, axis_Y, -38.66) ##first rotate y -38.66 then rotate x 45
    transform = transform[:3, :3]
    return transform.T

def trans1_9():
    transform = np.identity(4)
    transform = Rotate(transform, axis_Y, 38.66)
    transform = Rotate(transform, axis_X, -45)
    transform = transform[:3, :3]
    return transform.T


def trans9_2():
    transform = np.identity(4)
    transform=Rotate(transform,axis_X,45)
    transform = Rotate(transform, axis_Y, 38.66) ##first rotate y -38.66 then rotate x 45
    transform = transform[:3, :3]
    return transform.T

def trans2_9():
    transform = np.identity(4)
    transform = Rotate(transform, axis_Y,-38.66)
    transform = Rotate(transform, axis_X, -45)
    transform = transform[:3, :3]
    return transform.T

def trans9_3():
    transform = np.identity(4)
    transform=Rotate(transform,axis_X,45)
    transform = Rotate(transform, axis_Y, 51.34)
    transform = transform[:3, :3]
    return transform.T

def trans3_9():
    transform = np.identity(4)
    transform = Rotate(transform, axis_Y, -51.34)
    transform = Rotate(transform, axis_X, -45)
    transform = transform[:3, :3]
    return transform.T

def trans9_4():
    transform = np.identity(4)
    transform=Rotate(transform,axis_X,45)
    transform = Rotate(transform, axis_Y, 128.66)
    transform = transform[:3, :3]
    return transform.T

def trans4_9():
    transform = np.identity(4)
    transform = Rotate(transform, axis_Y, -128.66)
    transform = Rotate(transform, axis_X, -45)
    transform = transform[:3, :3]
    return transform.T

def trans9_5():
    transform = np.identity(4)
    transform=Rotate(transform,axis_X,45)
    transform = Rotate(transform, axis_Y, 141.34)
    transform = transform[:3, :3]
    return transform.T

def trans5_9():
    transform = np.identity(4)
    transform = Rotate(transform, axis_Y, -141.34)
    transform = Rotate(transform, axis_X, -45)
    transform = transform[:3, :3]
    return transform.T

def trans9_6():
    transform = np.identity(4)
    transform=Rotate(transform,axis_X,45)
    transform = Rotate(transform, axis_Y, -141.34)
    transform = transform[:3, :3]
    return transform.T

def trans6_9():
    transform = np.identity(4)
    transform = Rotate(transform, axis_Y, 141.34)
    transform = Rotate(transform, axis_X, -45)
    transform = transform[:3, :3]
    return transform.T

def trans9_7():
    transform = np.identity(4)
    transform=Rotate(transform,axis_X,45)
    transform = Rotate(transform, axis_Y, -128.66)
    transform = transform[:3, :3]
    return transform.T

def trans7_9():
    transform = np.identity(4)
    transform = Rotate(transform, axis_Y, 128.66)
    transform = Rotate(transform, axis_X, -45)
    transform = transform[:3, :3]
    return transform.T

## the list of transform matrix
inv_transform_list=[trans0_9(),trans1_9(),trans2_9(),trans3_9(),trans4_9(),trans5_9(),trans6_9(),
                trans7_9(),trans8_9(),np.identity(3),trans10_9(),trans11_9(),trans12_9()]
transform_list=[trans9_0(),trans9_1(),trans9_2(),trans9_3(),trans9_4(),trans9_5(),
                    trans9_6(),trans9_7(),trans9_8(),np.identity(3),trans9_10(),trans9_11(),trans9_12()]

'''
def main():
    mat1 = np.array([[1.0, 0.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0],
                 [0.0, 0.0, 0.0, 1.0]],dtype=np.float32)
    p=np.array([0.0,0.0,1.0],dtype=np.float32)
    center=np.array([9,6,3])
    eye=np.array([8,5,-3])
    up=np.array([0,1,0])

    #mat2=LookAt(eye,center,up)
    mat2=Perspective(45.0, 640.0/480.0, 0.1, 100.0)
    axis=np.array([0.0,0.0,1.0])
    mat3=Rotate_orginal(mat1,axis,30)
    #print mat1.shape[1]

    print mat3

if __name__ == '__main__':
    main()
    '''
