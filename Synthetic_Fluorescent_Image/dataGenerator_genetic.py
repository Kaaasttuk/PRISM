import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/Users/lirui/Downloads/IIB-Project/')

from Synthetic_Data_Generation.cell import Cell
from Synthetic_Data_Generation.microscope import Microscope
from utils import normalize_image, get_noisy_background, segment_synthetic_cell

from tqdm.auto import tqdm
import numpy as np
import random
import pickle
from skimage.util import random_noise


all_channel_names = ['BFP', 'YFP', 'RFP']

waveLengths = {
    'BFP': 460,
    'YFP': 540,
    'RFP': 620
}

l_range = [500,2000] #[1100,2800]#   #nm  
outer_r_range = [450,510] #nm
trench_width = 1500 #nm
periplasm_thickness_range = [40,50] #nm


p_gen = 30 #nm/px
p_real = 65 #nm/px
image_size_real = np.array([64,36]) #[48,20]
image_size = np.round(image_size_real*p_real).astype(int)  #np.array([5000, 1.5*trench_width]) #nm
image_size_gen = np.round(image_size/p_gen).astype(int)# image_size_gen = np.round(image_size/p_gen).astype(int)


brightness_cutoff = 1.0
camera_bias = 0.2
alpha = 1 #multiplicative gain
gauss_noise_sigma = 0.005


nucleoid_types = ['spherocylinder','twin-spherocylinder']
NA = 1.3
refractive_index = 1.515
use_rect_mask = True
mask_centerloc = [0.5,0.5]

# for empty background generation
noise_mu = 0.5
noise_sigma = 0.14


N = 1000

all_images = np.zeros((N, image_size_real[0], image_size_real[1], 1))#, dtype=np.uint16)
all_labels = np.zeros((N,1))

for idx in tqdm(range(N)):
    channel_name = np.random.choice(all_channel_names, 1, p=[0.3,0.3,0.4])[0]
    #print(channel_name)

    waveLength = waveLengths[channel_name]
    centerloc = [random.gauss(0.5, 0.03), random.gauss(0.5, 0.03)] # Shift the mask around the cell
    for i in range(len(centerloc)):
        if centerloc[i]<0.4:
            centerloc[i]=0.4
        elif centerloc[i]>0.6:
            centerloc[i]=0.6
    centerloc = [0.5,0.5]
    

    outer_r = random.randint(outer_r_range[0], outer_r_range[1])
    l = random.randint(l_range[0], l_range[1])
    
    periplasm_thickness = random.randint(periplasm_thickness_range[0], periplasm_thickness_range[1])
    

    label = np.random.choice(np.arange(0,8),1)[0] 
    '''Label: 
        0: BG
        1: membrane only
        2: nucleoid only
        3: IDP only
        4: M+N
        5: M+IDP
        6: N+IDP
        7: M+N+IDP
    '''
    if label in [3,5,6,7]:
        with_IDP = True
    else:
        with_IDP = False

    all_labels[idx] = label


    nucleoid_type = np.random.choice(nucleoid_types,1, p=[0.8,0.2])[0] 

    cell = Cell(l, outer_r, periplasm_thickness, nucleoid_type, p_gen, image_size_gen, brightness_cutoff, trench_width, with_IDP)
    cell = cell.generate_3D_cell()
    rotation = np.random.uniform(-cell.get_max_rotation_angle(), cell.get_max_rotation_angle(), 1)[0] #in degree

    
    # microscope model
    ch_microscope = Microscope(p_real, waveLength, NA, refractive_index)

    # cell model      
    objects3D = []
    intensities = []
    cutoff_options = []

    if label != 0:
        if label == 1:
            molecules_in_periplasm = cell.sample_from_component(cell.periplasm,1.2)
            objects3D.append(molecules_in_periplasm)
            intensities.append(1.0)
            cutoff_options.append(0)

        elif label==2:
            molecules_in_nucleoid = cell.sample_from_component(cell.nucleoid, 0.2)
            objects3D.append(molecules_in_nucleoid)
            intensities.append(0.5)
            cutoff_options.append(0)
            
        elif label==3:
            molecules_in_IDP = cell.sample_from_component(cell.IDP, 1.0)
            objects3D.append(molecules_in_IDP)
            intensities.append(0.8)
            cutoff_options.append(0)

            molecules_in_cytoplasm = cell.sample_from_component(cell.cytoplasm, 0.5)
            objects3D.append(molecules_in_cytoplasm)
            intensities.append(0.5)
            cutoff_options.append(0)
        
        elif label==4:
            molecules_in_periplasm = cell.sample_from_component(cell.periplasm,1.2)
            objects3D.append(molecules_in_periplasm)
            intensities.append(0.9)
            cutoff_options.append(0)

            molecules_in_nucleoid = cell.sample_from_component(cell.nucleoid, 1.0)
            objects3D.append(molecules_in_nucleoid)
            intensities.append(0.4)
            cutoff_options.append(0)

        elif label == 5:
            molecules_in_periplasm = cell.sample_from_component(cell.periplasm,1.2)
            objects3D.append(molecules_in_periplasm)
            intensities.append(0.8)
            cutoff_options.append(0)

            molecules_in_IDP = cell.sample_from_component(cell.IDP, 1.0)
            objects3D.append(molecules_in_IDP)
            intensities.append(0.2)
            cutoff_options.append(0)

        
        elif label == 6:
            molecules_in_nucleoid = cell.sample_from_component(cell.nucleoid, 1.0)
            objects3D.append(molecules_in_nucleoid)
            intensities.append(0.5)
            cutoff_options.append(0)

            molecules_in_IDP = cell.sample_from_component(cell.IDP, 1.0)
            objects3D.append(molecules_in_IDP)
            intensities.append(0.4)
            cutoff_options.append(0)

            molecules_in_cytoplasm = cell.sample_from_component(cell.cytoplasm, 0.1)
            objects3D.append(molecules_in_cytoplasm)
            intensities.append(0.1)
            cutoff_options.append(0)

        elif label == 7:

            molecules_in_periplasm = cell.sample_from_component(cell.periplasm,1.2)
            objects3D.append(molecules_in_periplasm)
            intensities.append(0.8)
            cutoff_options.append(0)

            molecules_in_nucleoid = cell.sample_from_component(cell.nucleoid, 1.0)
            objects3D.append(molecules_in_nucleoid)
            intensities.append(0.2)
            cutoff_options.append(0)

            molecules_in_IDP = cell.sample_from_component(cell.IDP, 1.0)
            objects3D.append(molecules_in_IDP)
            intensities.append(0.2)
            cutoff_options.append(0)

            # include neighbour
            # molecules_in_neighbour = cell.sample_from_component(cell.neighbour, 0.3)
            # objects3D.append(molecules_in_neighbour)
            # intensities.append(0.3)
            # cutoff_options.append(0)


        channel_image = cell.get_greyScale_image(objects3D, intensities, cutoff_options, rotation, centerloc) #float image
        blurred_image = ch_microscope.convolve_with_Gaussian(channel_image, cell.p_gen)
        resized = ch_microscope.resize_image(blurred_image, image_size_real)

        noisy_resized_poisson = random_noise(resized, mode ='poisson') # add photon shot noise. Input is float image. 
        noisy_resized = 1*noisy_resized_poisson + camera_bias
        noisy_resized_gaussian = random_noise(noisy_resized, mode ='gaussian', mean = 0, var = gauss_noise_sigma**2) # add thermal noise
        noisy_resized = normalize_image(noisy_resized_gaussian)

        # Segment the cell out

        input_image = segment_synthetic_cell(normalize_image(noisy_resized), cell, rotation, use_rect_mask, mask_centerloc, erode=False, no_rotation=False) # always returns an image of type float
        #input_image = normalize_image(noisy_resized)
        all_images[idx,:,:,0] = input_image
    else:
        noisy_resized = get_noisy_background(image_size_real, 1, noise_mu, noise_sigma)
        
        input_image = segment_synthetic_cell(normalize_image(noisy_resized), cell, rotation, use_rect_mask, mask_centerloc, erode=True, no_rotation=False) # always returns an image of type float
        #input_image = normalize_image(noisy_resized)
        all_images[idx,:,:,0] = input_image


training_data = (all_images, all_labels)

pickle.dump(training_data, open("/Users/lirui/Downloads/IIB-Project/ML_model/ch_training_data/experimental_data.pkl", "wb" ) )