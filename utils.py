import random
from flyingcircus_numeric.base import normalize
import numpy as np
import mahotas as mh
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from skimage.transform import resize
from skimage import data, img_as_float, img_as_ubyte, color

def normalize_image(image):
    normalized = (image-np.min(image))/(np.max(image)-np.min(image))
    return normalized
    
def get_noisy_background(image_size, amount_of_noise, mu, sigma):
    N = image_size[0]*image_size[1]
    image = np.zeros((N,))
    M = int(amount_of_noise*N)
    all_indices = np.arange(0,N)
    selected_indices = np.random.choice(all_indices, M, False)
    values = mu*np.ones(M) + sigma*np.random.randn(M)
    image[selected_indices] = values
    image = image.reshape((image_size[0], image_size[1]))
    return image

def generate_label(num_of_channels, M, one_type_each_channel, one_channel): 
# M is the number of types for each channel; one_channel means there is only fluorescent barcode in one imaging channel -- only applicable when num_of_channels > 1
    if one_type_each_channel:
        label = np.zeros((num_of_channels,M), dtype=np.uint64)
        if one_channel:
            i = np.random.randint(0,num_of_channels-1)
            j = random.randint(0,M-1)
            label[i,j] = 1
        else:
            for i in range(num_of_channels):
                j = random.randint(0,M-1)
                label[i,j] = 1
    else:
        if one_channel:
            label = np.zeros((num_of_channels,M), dtype=np.uint64)
            i = np.random.randint(0,num_of_channels-1)
            channel_label = np.random.choice([0,1], size=(M), replace=True).reshape((1,M))
            label[i,:] = channel_label
        else:
            label = np.random.choice([0,1], size=(num_of_channels*M), replace=True).reshape((num_of_channels,M))
    return label


def segment_synthetic_cell(img, cell, rotation, rectangular, centerloc, erode=False, no_rotation=False):
    if rectangular==False:
        h,w = img.shape
        img_size = [h,w]

        mask = np.zeros((cell.image_size_gen[0], cell.image_size_gen[1]), dtype=np.float64)
        projected = np.sum(cell.cell,axis=1)
        isCell = img_as_ubyte(projected>0)

        mask_height,mask_width = isCell.shape
        centerloc = np.array(centerloc)
        topleftloc = np.round(np.multiply(centerloc,cell.image_size_gen)-0.5*np.array(isCell.shape)).astype(int)
        mask[topleftloc[0]:topleftloc[0]+mask_height,topleftloc[1]:topleftloc[1]+mask_width] = isCell

        #Rotate first, then resize
        PIL_img = Image.fromarray(mask)
        rotated_PIL_mask = PIL_img.rotate(rotation)
        rotated_mask = np.asarray(rotated_PIL_mask)

        rotated_mask = rotated_mask>0
        rotated_mask = img_as_ubyte(rotated_mask)

        dilated_mask = mh.morph.dilate(rotated_mask)
        dilated_mask = mh.morph.dilate(dilated_mask)
        dilated_mask = mh.morph.dilate(dilated_mask)

        final_mask = resize(dilated_mask, img_size ,anti_aliasing=True)>0
    else:
        img_h,img_w = img.shape
        img_size = [img_h,img_w]

        h, w1, w2 =  cell.cell.shape
        if erode == True:
            # h = int(0.86*h)
            # w1 = int(0.86*w1)
            h = int(0.95*h)
            w1 = int(0.95*w1)

        mask = np.zeros((cell.image_size_gen[0], cell.image_size_gen[1]), dtype=np.float64)
        topleftloc = np.round(np.multiply(centerloc, cell.image_size_gen)-0.5*np.array([h,w1])).astype(int)
        mask[topleftloc[0]:topleftloc[0]+h,topleftloc[1]:topleftloc[1]+w1] = 1

        resized_mask = resize(mask, img_size, anti_aliasing=True)>0

        if erode == False:
            new_mask = mh.morph.dilate(resized_mask)
        else:
            new_mask = mh.morph.erode(resized_mask)
            new_mask = mh.morph.erode(resized_mask)

        new_mask = img_as_ubyte(new_mask)

        PIL_img = Image.fromarray(new_mask)
        rotated_PIL_mask = PIL_img.rotate(rotation, resample = Image.BILINEAR)
        final_mask = np.asarray(rotated_PIL_mask)>0
        
        if no_rotation == True:
            max_y = np.max(np.where(final_mask>0)[0])
            min_y = np.min(np.where(final_mask>0)[0])
            max_x = np.max(np.where(final_mask>0)[1])
            min_x = np.min(np.where(final_mask>0)[1])
            new_final_mask = np.zeros(final_mask.shape)
            new_final_mask[min_y:(max_y+1), min_x:(max_x+1)] = 1
            final_mask = (new_final_mask>0)

    selected_vec = img[final_mask]
    min_val = np.min(selected_vec)
    max_val = np.max(selected_vec)
    img[final_mask] = (selected_vec-min_val)/(max_val-min_val)
    seg_synthetic = img*final_mask
    return seg_synthetic # np.float

def raw_moment(data, iord, jord):
    nrows, ncols = data.shape
    y, x = np.mgrid[:nrows, :ncols]
    data = data * x**iord * y**jord
    return data.sum()

def moments_cov(data):
    data_sum = data.sum()
    m10 = raw_moment(data, 1, 0)
    m01 = raw_moment(data, 0, 1)
    x_centroid = m10 / data_sum
    y_centroid = m01 / data_sum
    u11 = (raw_moment(data, 1, 1) - x_centroid * m01) / data_sum
    u20 = (raw_moment(data, 2, 0) - x_centroid * m10) / data_sum
    u02 = (raw_moment(data, 0, 2) - y_centroid * m01) / data_sum
    cov = np.array([[u20, u11], [u11, u02]])
    return cov

# def get_rectangular_mask(mc_mask): 
#     '''
#         Input: 
#             mc_mask: contains the segmented mother cell (convexhull, pre dilation)
#         Output: 
#             rectangular_mc_mask: a rectangular fitted to the mother cell
#     '''
#     img = mc_mask.copy()
#     y,x = np.nonzero(img)
#     centroid = (np.mean(x),np.mean(y))
#     x = x - np.mean(x)
#     y = y - np.mean(y)
#     coords = np.vstack([x, y])

#     cov = moments_cov(img)
#     evals, evecs = np.linalg.eig(cov)

#     sort_indices = np.argsort(evals)[::-1]
#     x_v1, y_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
#     x_v2, y_v2 = evecs[:, sort_indices[1]]
#     v1 = np.array([x_v1, y_v1])
#     v2 = np.array([x_v2, y_v2])

#     coords_mat = np.stack((x,y), axis=1)
#     l_v1 = np.sign(y_v1)*np.matmul(coords_mat, v1)
#     l_v2 = np.sign(x_v2)*np.matmul(coords_mat, v2)

#     min_x = np.round(np.min(l_v2))
#     max_x = np.round(np.max(l_v2))
#     min_y = np.round(np.min(l_v1))
#     max_y = np.round(np.max(l_v1))
#     width = int(max_x-min_x+1)
#     height = int(max_y-min_y+1)
#     theta = np.arctan((x_v1)/(y_v1))  #anti-clockwise is positive
#     # Rotate
#     rect_img = np.ones((height,width))
#     rect_img = Image.fromarray(rect_img)
#     rotated = np.array(rect_img.rotate(np.degrees(theta), resample = Image.BILINEAR, expand=True))
#     rotated = rotated>0
#     new_height, new_width = rotated.shape

#     #find where to put this rectangle
#     anchor = [-min_x, -min_y] #anchor has to be placed at the centroid
#     center_to_anchor = [anchor[0]-width/2, anchor[1]-height/2] 
#     rotation_mat = np.matrix([[np.cos(theta), np.sin(theta)],
#                         [-np.sin(theta), np.cos(theta)]])

#     new_center_to_anchor = rotation_mat * (np.array(center_to_anchor).reshape((2,1)))
#     new_center_to_anchor = np.array(new_center_to_anchor).flatten()
#     new_center_pos = [centroid[0]-new_center_to_anchor[0], centroid[1]-new_center_to_anchor[1]] #[x,y]
#     left_top_pos = [int(new_center_pos[0]- new_width/2), int(new_center_pos[1]-new_height/2)] #[x,y]

#     mc_mask = np.zeros(img.shape)
#     mc_mask[left_top_pos[1]:left_top_pos[1]+rotated.shape[0], left_top_pos[0]:left_top_pos[0]+rotated.shape[1]]=rotated
#     mc_mask = mc_mask==1

#     return mc_mask
def get_rectangular_mask(mc_mask):  
    '''
        Input: 
            mc_mask: contains the segmented mother cell (convexhull, pre dilation)
        Output: 
            rectangular_mc_mask: a rectangular fitted to the mother cell
    '''
    img = mc_mask.copy()
    y, x = np.nonzero(img)
    centroid = (np.mean(x), np.mean(y))
    x = x - np.mean(x)
    y = y - np.mean(y)
    coords = np.vstack([x, y])

    cov = moments_cov(img)
    evals, evecs = np.linalg.eig(cov)

    sort_indices = np.argsort(evals)[::-1]
    x_v1, y_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
    x_v2, y_v2 = evecs[:, sort_indices[1]]
    v1 = np.array([x_v1, y_v1])
    v2 = np.array([x_v2, y_v2])

    coords_mat = np.stack((x, y), axis=1)
    l_v1 = np.sign(y_v1) * np.matmul(coords_mat, v1)
    l_v2 = np.sign(x_v2) * np.matmul(coords_mat, v2)

    min_x = np.round(np.min(l_v2))
    max_x = np.round(np.max(l_v2))
    min_y = np.round(np.min(l_v1))
    max_y = np.round(np.max(l_v1))
    width = int(max_x - min_x + 1)
    height = int(max_y - min_y + 1)
    theta = np.arctan((x_v1) / (y_v1))  # anti-clockwise is positive

    # Rotate
    rect_img = np.ones((height, width))
    rect_img = Image.fromarray(rect_img)
    rotated = np.array(rect_img.rotate(np.degrees(theta), resample=Image.BILINEAR, expand=True))
    rotated = rotated > 0
    new_height, new_width = rotated.shape

    # Ensure that the rotated image fits within mc_mask
    if new_height > img.shape[0] or new_width > img.shape[1]:
        new_height = min(new_height, img.shape[0])
        new_width = min(new_width, img.shape[1])
        rotated = rotated[:new_height, :new_width]

    # Find where to put this rectangle
    anchor = [-min_x, -min_y]  # anchor has to be placed at the centroid
    center_to_anchor = [anchor[0] - width / 2, anchor[1] - height / 2]
    rotation_mat = np.matrix([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])

    new_center_to_anchor = rotation_mat * (np.array(center_to_anchor).reshape((2, 1)))
    new_center_to_anchor = np.array(new_center_to_anchor).flatten()
    new_center_pos = [centroid[0] - new_center_to_anchor[0], centroid[1] - new_center_to_anchor[1]]  # [x, y]
    left_top_pos = [int(new_center_pos[0] - new_width / 2), int(new_center_pos[1] - new_height / 2)]  # [x, y]

    # Ensure left_top_pos stays within bounds
    left_top_pos[0] = max(0, min(left_top_pos[0], img.shape[1] - new_width))
    left_top_pos[1] = max(0, min(left_top_pos[1], img.shape[0] - new_height))

    mc_mask = np.zeros(img.shape)
    mc_mask[left_top_pos[1]:left_top_pos[1] + rotated.shape[0], left_top_pos[0]:left_top_pos[0] + rotated.shape[1]] = rotated
    mc_mask = mc_mask == 1

    return mc_mask

def place_img_on_background(object_img, background_img, centerloc):
    object_size = np.array(object_img.shape)
    background_size = np.array(background_img.shape)

    bg_topleftloc = np.round(np.multiply(centerloc,background_size)-0.5*np.array(object_img.shape)).astype(int)
    ob_topleftloc = [0,0]
    ob_bottomrightloc = object_size
    for i in range(2):
        if bg_topleftloc[i]<0:
            ob_topleftloc[i] = -bg_topleftloc[i]
            bg_topleftloc[i] = 0
        elif bg_topleftloc[i] > (background_size[i]-object_size[i]):
            ob_bottomrightloc[i] -= (bg_topleftloc[i]+object_size[i]-background_size[i])

    cropped_object_img = object_img[ob_topleftloc[0]:ob_bottomrightloc[0], ob_topleftloc[1]:ob_bottomrightloc[1]]
    cropped_object_size = np.array(cropped_object_img.shape)

    background_img[bg_topleftloc[0]:bg_topleftloc[0]+cropped_object_size[0], bg_topleftloc[1]:bg_topleftloc[1]+cropped_object_size[1]] = cropped_object_img
    
    return background_img

def get_Butterworth_filter(M,N, D0, n): # M: img height; N: img width; [M,N]=img.shape; D0: cutoff frequency; n: order
    u = np.arange(0,M)
    v = np.arange(0,N)
    idx = np.where(u>M/2)
    u[idx] = u[idx]-M
    idy = np.where(v>N/2)
    v[idy] = v[idy]-N
    [V,U] = np.meshgrid(v,u)

    D = np.sqrt((U*N/M)**2 + V**2)
           
    H = 1/np.sqrt(1 + (D/D0)**(2*n))
    return H

def filter_with_Butterworth(img, H):
    img = normalize_image(img)
    FT_img = np.fft.fft2(img)
    G = np.multiply(H, FT_img)
    output_image = np.fft.ifft2(G).real
    return output_image


if __name__ == "__main__":
    noisy_background = get_noisy_background([64,32], 1, 0.5, 0.14)
    plt.imshow(noisy_background)
    plt.show()

    print('finish')