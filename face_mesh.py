import cv2
import mediapipe as mp
from typing import List, Tuple, Dict
import numpy

def get_connections(valid_point_indices: List[int]) -> List[Tuple[int,int]]:
    """
    Given a list of valid mesh indices, it will return a list of tuples representing their connections.
    Args:
        `valid_point_indices`: List of landmark indices
    
    Returns:
        A list of tuples indicating the connections between the given landmarks.
    """

    face_mesh = mp.solutions.face_mesh
    all_connections = face_mesh.FACEMESH_TESSELATION
    valid_connections = []
    for i,j in all_connections:
        if i in valid_point_indices and j in valid_point_indices:
            valid_connections.append((i,j))
    return valid_connections

def get_binary_mask(image: numpy.ndarray, threshold:int=128) -> numpy.ndarray:
    """
    Converts an image into a black and white binary mask {0,255} given a threshold.
    Args:
        `image`: The input image in grayscale or in BGR
        `threshold`: All the intensities lower than the threshold become 0, whereas all
        the intensities greater than or equal to the threhold become 255.
        
    Returns:
        Binary grayscale image.
    """
    if image.shape[2] > 1:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    mask = image.copy()
    mask[mask< threshold] = 0
    mask[mask>=threshold] = 255
    return mask

def get_face_region_mesh(image: numpy.ndarray, face_region_mask: numpy.ndarray) -> Tuple[List[int], List[Tuple[int,int]]]:
    """
    Given an image and a mask denoting a face region, it returns the indices and the connections of landmarks inside that region.
    Args:
        `image`: Input face image
        `face_region_mask`: Binary mask of a face region
    
    Returns:
        Indices and connections of the face mesh landmarks inside the requested face region
    """
    
    face_mesh_solver = mp.solutions.face_mesh.FaceMesh()

    img_w = image.shape[1]
    img_h = image.shape[0]
    face_mesh_results = list(face_mesh_solver.process(image).multi_face_landmarks[0].landmark)
    region_indices = []
    
    for idx in range(468):
        x,y = int(face_mesh_results[idx].x*img_w),int(face_mesh_results[idx].y*img_h)
        if face_region_mask[y,x] != 0:
            region_indices.append(idx)

    region_connections = get_connections(set(region_indices))
    return region_indices, region_connections

def find_landmark_coordinates(image: numpy.ndarray, landmark_indices: List[int]) -> Dict[int, Tuple[int,int]]:
    """
    Returns the x,y coordinates of the given landmarks with respect to the input image.
    Args:
        `image`: Input image
        `landmark_indices`: List of landmark indices

    Returns:
        A dictionary that maps each index to the x,y coordinates
    """
    face_mesh_solver = mp.solutions.face_mesh.FaceMesh()
    face_mesh_results = list(face_mesh_solver.process(image).multi_face_landmarks[0].landmark)
    
    img_w = image.shape[1]
    img_h = image.shape[0]
    index2point = {}
    for idx in landmark_indices:
        x,y = int(face_mesh_results[idx].x*img_w), int(face_mesh_results[idx].y*img_h)
        index2point[idx] = (x,y)

    return index2point

if __name__ == "__main__":
    # In this example, we first load a reference image with two different region masks.
    # Then we find those regions on another arbitrary face

    # load reference image
    refimg = cv2.imread('average_face.jpeg')

    # load region masks
    regions = [cv2.imread('average_face_mask1.jpg'), cv2.imread('average_face_mask2.jpg')]

    # convert regions to masks
    masks = [get_binary_mask(region, 2) for region in regions]

    # get landmark indices and connections
    regions_mesh = [get_face_region_mesh(refimg, mask) for mask in masks]

    # load a test image
    testimg = cv2.imread('test.jpg')

    # find landmark coordinates in the test image for each region
    regions_mesh_coord = [find_landmark_coordinates(testimg, mesh[0]) for mesh in regions_mesh]

    # draw the region mesh on the test image and save
    for region_idx, (mesh, coord) in enumerate(zip(regions_mesh, regions_mesh_coord)):
        testimgcopy = testimg.copy()

        for idx in coord:
            cv2.circle(testimgcopy, coord[idx], 6, (100,100,0), -1)

        for i,j in mesh[1]:
            cv2.line(testimgcopy, coord[i], coord[j], (100,100,0), 2)

        cv2.imwrite(f'test_image_region_{region_idx}.jpg', testimgcopy)