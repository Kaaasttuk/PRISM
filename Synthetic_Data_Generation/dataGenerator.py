import sys
sys.path.insert(1, "D:\PRISM")

from Synthetic_Data_Generation.cell import Cell
from Synthetic_Data_Generation.microscope import Microscope
from utils import normalize_image, get_noisy_background, generate_label, segment_synthetic_cell, get_Butterworth_filter

from tqdm.auto import tqdm
import numpy as np
import random
import pickle
from skimage import img_as_ubyte
from skimage.util import random_noise

# Can add more channels and corresponding information
# all_channel_names = ['BFP', 'GFP', 'RFP', 'CY5']
all_channel_names = ['BFP', 'GFP', 'RFP']

waveLengths = {
    'BFP': 460,
    'GFP': 535,
    'RFP': 620,
    'CY5': 800
}

lumMatch_parameters = { # images in uint8. Obtained from normalized sample images.
    'BFP': {'M':83, 'S':61},
    'GFP': {'M':102, 'S':44},
    'RFP': {'M':80, 'S':52},
    'CY5': {'M':39, 'S':54}
}

BG_lumMatch_parameters = { # images in uint8
    'BFP': {'M':123, 'S':36},
    'GFP': {'M':125, 'S':35},
    'RFP': {'M':129, 'S':36},
    'CY5': {'M':114, 'S':47}
}

locations = ['membrane', 'nucleoid']
one_type_each_channel = False
num_of_locs = len(locations)
l_range = [500,2000]#[1100,2800]#nm  
outer_r_range = [450,510] #nm
trench_width = 1500 #nm
periplasm_thickness_range = [40,50] #nm


p_gen = 30 #nm/px
p_real = 65 #nm/px
image_size_real = np.array([64,36])
image_size = np.round(image_size_real*p_real).astype(int)  #np.array([5000, 1.5*trench_width]) #nm
image_size_gen = np.round(image_size/p_gen).astype(int)# image_size_gen = np.round(image_size/p_gen).astype(int)


brightness_cutoff = 0.3
camera_bias = 0.2
alpha = 1 #multiplicative gain
gauss_noise_sigma = {
    'BFP': 0.007,
    #'CFP': 0.018,
    'GFP': 0.02,
    'RFP': 0.003,
    'CY5': 0.003
}
peri_sampling_proportion = {
    'BFP': 0.2,
    'GFP': 0.8,
    'RFP': 0.15,
    'CY5': 0.2
}


nucleoid_types = ['spherocylinder','twin-spherocylinder']
NA = 1.3
refractive_index = 1.515
use_rect_mask = True
mask_centerloc = [0.5,0.5]

# for empty background generation
noise_mu = 0.5
noise_sigma = 0.14


N = 100 # total number of image-label pairs

all_images = np.zeros((N, image_size_real[0], image_size_real[1], 1))#, dtype=np.uint16)
all_labels = np.zeros((N,1))

for idx in tqdm(range(N)):
    # A total of N=100 synthetic images are created. In each loop iteration:
    # A random channel (BFP, GFP, RFP, CY5) is selected based on predefined probabilities.
    channel_name = np.random.choice(all_channel_names, 1, p=[0.3,0.3,0.39,0.01])[0]
    #print(channel_name)

    # Random parameters are assigned to the synthetic cell, such as cell length (l), outer radius (outer_r), and fluorescence properties.
    waveLength = waveLengths[channel_name]
    centerloc = [random.gauss(0.5, 0.03), random.gauss(0.5, 0.03)] # Shift the mask around the cell
    for i in range(len(centerloc)):
        if centerloc[i]<0.4:
            centerloc[i]=0.4
        elif centerloc[i]>0.6:
            centerloc[i]=0.6

    outer_r = random.randint(outer_r_range[0], outer_r_range[1])
    l = random.randint(l_range[0], l_range[1])
    
    periplasm_thickness = random.randint(periplasm_thickness_range[0], periplasm_thickness_range[1])
    
    label = np.random.choice([0,1,2,3],1)[0] # The label of the synthetic cell is randomly selected: 0 is BG, 1 is membrane only, 2 is nucleoid only, 3 is both

    if channel_name == 'CY5':
        label = 2

    all_labels[idx] = label


    nucleoid_type = np.random.choice(nucleoid_types,1, p=[0.8,0.2])[0] 

    # A synthetic cell is generated with random attributes using the Cell class. The cell is rotated randomly to simulate variability in its orientation.

    cell = Cell(l, outer_r, periplasm_thickness, nucleoid_type, p_gen, image_size_gen, brightness_cutoff, trench_width,False)
    cell = cell.generate_3D_cell()
    rotation = np.random.uniform(-cell.get_max_rotation_angle(), cell.get_max_rotation_angle(), 1)[0] #in degree

    # microscope model simulates how the generated 3D cell would be imaged under specific optical settings
    ch_microscope = Microscope(p_real, waveLength, NA, refractive_index)
    # cell model      
    objects3D = []
    intensities = []
    cutoff_options = []

    
    if label!=0:
        #print(label)
        if label==1:
            molecules_in_periplasm = cell.sample_from_component(cell.periplasm, peri_sampling_proportion[channel_name])
            objects3D.append(molecules_in_periplasm)
            intensities.append(0.2)#np.random.choice([0.4,0.5],1)[0])
            cutoff_options.append(1)

        elif label==3:
            molecules_in_periplasm = cell.sample_from_component(cell.periplasm, peri_sampling_proportion[channel_name])
            objects3D.append(molecules_in_periplasm)
            intensities.append(0.3)
            cutoff_options.append(1)

            molecules_in_nucleoid = cell.sample_from_component(cell.nucleoid, 0.5)
            objects3D.append(molecules_in_nucleoid)
            if channel_name== 'RFP':
                intensities.append(0.14)
            else:
                intensities.append(0.18)
            cutoff_options.append(0)

        else: #label==2
            molecules_in_nucleoid = cell.sample_from_component(cell.nucleoid, 0.5)
            objects3D.append(molecules_in_nucleoid)
            
            if channel_name == 'CY5':
                intensities.append(1.0)
            else:
                intensities.append(0.3)  #np.random.choice([0.3,1.0],1)[0])   #
            cutoff_options.append(0)
        
        # include part of neighbouring cell randomly
        include_neighbour = np.random.choice([0,1], 1, p=[0.7,0.3])[0]
        if include_neighbour==1: 
            #neighbour_option = np.random.choice([0,1,2],1,p=[0.5,0.2,0.3])[0] # 0 is upper neighbour only; 1 is lower neighbour only; 2 is both
            neighbour_option = np.random.choice([0,1,2],1,p=[1.0,0.0,0.0])[0]
            neighbour_intensity = 0.1

            if neighbour_option == 0:
                molecules_in_neighbour = cell.sample_from_component(cell.upper_neighbour, 0.3)
                objects3D.append(molecules_in_neighbour)
                intensities.append(neighbour_intensity)
                cutoff_options.append(0)
            elif neighbour_option == 1:
                molecules_in_neighbour = cell.sample_from_component(cell.lower_neighbour, 0.3)
                objects3D.append(molecules_in_neighbour)
                intensities.append(neighbour_intensity)
                cutoff_options.append(0)
            else:
                molecules_in_neighbour = cell.sample_from_component(cell.upper_neighbour, 0.3)
                objects3D.append(molecules_in_neighbour)
                intensities.append(neighbour_intensity)
                cutoff_options.append(0)

                molecules_in_neighbour = cell.sample_from_component(cell.lower_neighbour, 0.3)
                objects3D.append(molecules_in_neighbour)
                intensities.append(neighbour_intensity)
                cutoff_options.append(0)

        
        # Image and Noise Generation
        channel_image = cell.get_greyScale_image(objects3D, intensities, cutoff_options, rotation, centerloc) #float image
        blurred_image = ch_microscope.convolve_with_Gaussian(channel_image, cell.p_gen)
        resized = ch_microscope.resize_image(blurred_image, image_size_real)

        # Adding Noise
        noisy_resized = random_noise(resized, mode ='poisson') # add photon shot noise. Input is float image.
        noisy_resized = alpha*noisy_resized + camera_bias
        noisy_resized = random_noise(noisy_resized, mode ='gaussian', mean = 0, var = gauss_noise_sigma[channel_name]**2) # add thermal noise
        noisy_resized = img_as_ubyte(normalize_image(noisy_resized))

        # Normalization and Luminosity Matching
        M = lumMatch_parameters[channel_name]['M']
        S = lumMatch_parameters[channel_name]['S']
        lumMatched = ((noisy_resized-np.mean(noisy_resized))/np.std(noisy_resized)) * S + M
        lumMatched = lumMatched/255

        # Segment the cell out
        input_image = segment_synthetic_cell(normalize_image(lumMatched), cell, rotation, use_rect_mask, mask_centerloc, erode=True) # always returns an image of type float
        all_images[idx,:,:,0] = input_image
    else:
        BG_type = np.random.choice([0,1],1, p=[0.7,0.3]) # TODO
        #BG_type = 0
        if BG_type==0: #pure noise
            noisy_BG = get_noisy_background(image_size_real, 1, noise_mu, noise_sigma)
            noisy_BG = normalize_image(noisy_BG)
            noisy_BG = img_as_ubyte(noisy_BG)

            # lumMatch
            M = BG_lumMatch_parameters[channel_name]['M']
            S = BG_lumMatch_parameters[channel_name]['S']
            lumMatched = ((noisy_BG-np.mean(noisy_BG))/np.std(noisy_BG)) * S + M
            lumMatched = lumMatched/255

            input_image = segment_synthetic_cell(normalize_image(lumMatched), cell, rotation, use_rect_mask, mask_centerloc) # always returns an image of type float
        
        else: # include part of the neighbouring cell
            #neighbour_option = np.random.choice([0,1,2],1,p=[0.5,0.2,0.3])[0] # 0 is upper neighbour only; 1 is lower neighbour only; 2 is both
            neighbour_option = np.random.choice([0,1,2],1,p=[1.0,0.0,0.0])[0]   
            if neighbour_option == 0:
                molecules_in_neighbour = cell.sample_from_component(cell.upper_neighbour, 0.3)
                objects3D.append(molecules_in_neighbour)
                intensities.append(0.6)
                cutoff_options.append(0)
            elif neighbour_option == 1:
                molecules_in_neighbour = cell.sample_from_component(cell.lower_neighbour, 0.3)
                objects3D.append(molecules_in_neighbour)
                intensities.append(0.6)
                cutoff_options.append(0)
            else:
                molecules_in_neighbour = cell.sample_from_component(cell.upper_neighbour, 0.3)
                objects3D.append(molecules_in_neighbour)
                intensities.append(0.6)
                cutoff_options.append(0)

                molecules_in_neighbour = cell.sample_from_component(cell.lower_neighbour, 0.3)
                objects3D.append(molecules_in_neighbour)
                intensities.append(0.6)
                cutoff_options.append(0)


            channel_image = cell.get_greyScale_image(objects3D, intensities, cutoff_options, rotation, [0.5,0.5]) #float image  
            blurred_image = ch_microscope.convolve_with_Gaussian(channel_image, cell.p_gen)
            resized = ch_microscope.resize_image(blurred_image, image_size_real)

            noisy_resized = random_noise(resized, mode ='poisson') # add photon shot noise. Input is float image.
            noisy_resized = alpha*noisy_resized + camera_bias
            noisy_resized = random_noise(noisy_resized, mode ='gaussian', mean = 0, var = gauss_noise_sigma[channel_name]**2) # add thermal noise
            noisy_resized = normalize_image(noisy_resized)
            noisy_resized = img_as_ubyte(noisy_resized)

            # lumMatch
            M = lumMatch_parameters[channel_name]['M']
            S = lumMatch_parameters[channel_name]['S']
            lumMatched = ((noisy_resized-np.mean(noisy_resized))/np.std(noisy_resized)) * S + M
            lumMatched = lumMatched/255

            # Segment the cell out
            input_image = segment_synthetic_cell(normalize_image(lumMatched), cell, rotation, use_rect_mask, mask_centerloc, erode=True) # always returns an image of type float
        
        all_images[idx,:,:,0] = input_image

training_data = (all_images, all_labels)

# Saving Data
# Barcode_Screening was previously IIB_Project in Rui's Code
pickle.dump(training_data, open(r"D:\PRISM\Mother Cell For PRISM Train\Synthetic", "wb" ) )

