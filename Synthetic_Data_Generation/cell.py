import sys

from skimage.util import dtype
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/Users/lirui/Downloads/IIB-Project/')

import random
import numpy as np
from PIL import Image
import raster_geometry as rg
from utils import place_img_on_background

class Cell:
    def __init__(self, l, outer_r, periplasm_thickness, nucleoid_type, p_gen, image_size_gen, brightness_cutoff, trench_width, with_IDP=False):
        if (periplasm_thickness<outer_r):
            self.l = l #nm
            self.outer_r = outer_r #nm
            self.inner_r = outer_r-periplasm_thickness
            self.periplasm_thickness = periplasm_thickness #nm
        else:
            print("Your cell dimensions are not valid. Please reassign.")

        self.nucleoid_type_list = ['spherocylinder','twin-spherocylinder']
        if nucleoid_type in self.nucleoid_type_list:
            self.nucleoid_type = nucleoid_type
        else:
            print('Specified nucleoid type is unavailable! Available types are: %s' %{value for value in self.nucleoid_type_list})

        self.p_gen = p_gen #nm/px. default used for generation of the 3D cell model
        self.nucld_r = self.outer_r/np.sqrt(2)

        if self.l > 600:
            self.nucld_l = self.l/np.sqrt(2)
        else:
            self.nucld_l = 0.9*self.l

        # imaging
        self.trench_width = trench_width
        self.image_size_gen = image_size_gen
        # avoid bright lines
        self.brightness_cutoff = brightness_cutoff
        self.with_IDP = with_IDP

    def generate_3D_cell(self):
        self.l_gen = int(round(self.l/self.p_gen))
        self.outer_r_gen = int(round(self.outer_r/self.p_gen))
        self.inner_r_gen = min(self.outer_r_gen-1, int(round(self.inner_r/self.p_gen)))
        self.nucld_r_gen = int(self.nucld_r/self.p_gen)
        self.nucld_l_gen = int(self.nucld_l/self.p_gen)

        shape = [self.l_gen+2*self.outer_r_gen, 2*self.outer_r_gen, 2*self.outer_r_gen] # size of the rectangular box containing the cell
        outer_cylinder = rg.cylinder(
            shape = shape,
            height = self.l_gen,
            radius = self.outer_r_gen,
            axis=0,
            position=(0.5,0.5,0.5),
            smoothing=False)
        inner_cylinder = rg.cylinder(
            shape = shape,
            height = self.l_gen,
            radius = self.inner_r_gen,
            axis=0,
            position=(0.5,0.5,0.5),
            smoothing=False)
        outer_sphere1 = rg.sphere(shape,self.outer_r_gen,((self.outer_r_gen)/shape[0],0.5,0.5)) # top outer sphere
        outer_sphere2 = rg.sphere(shape,self.outer_r_gen,((shape[0] - self.outer_r_gen)/shape[0],0.5,0.5)) # bottom outer sphere
        inner_sphere1 = rg.sphere(shape,self.inner_r_gen,(self.outer_r_gen/shape[0],0.5,0.5)) # top inner sphere -- FP sphere (genetic barcode)
        inner_sphere2 = rg.sphere(shape,self.inner_r_gen,((shape[0] - self.outer_r_gen)/shape[0],0.5,0.5)) # bottom inner sphere -- FP sphere (genetic barcode)

        IDP_r_gen = int(self.inner_r_gen)
        
        upper_IDP = rg.sphere(shape,IDP_r_gen,((self.outer_r_gen-self.inner_r_gen+IDP_r_gen)/shape[0],0.5,0.5)) # top inner sphere -- FP sphere (genetic barcode)
        lower_IDP = rg.sphere(shape,IDP_r_gen,((shape[0] - (self.outer_r_gen-self.inner_r_gen+IDP_r_gen))/shape[0],0.5,0.5)) # bottom inner sphere -- FP sphere (genetic barcode)

        # Define the cell components
        self.cell = outer_cylinder+outer_sphere1+outer_sphere2
        self.periplasm = (outer_cylinder+outer_sphere1+outer_sphere2)^(inner_cylinder+inner_sphere1+inner_sphere2)
        self.cytoplasm = inner_cylinder+inner_sphere1+inner_sphere2
        IDP_choice = np.random.choice(['upper', 'lower'])
        if IDP_choice == 'upper':
            self.IDP = upper_IDP
        else:
            self.IDP = lower_IDP

        # Define neighbouring cell
        beta = random.uniform(0.9,1.3)
        upper_loc = random.choice([0.0,1.0])
        upper_neighbour = rg.sphere(shape,beta*self.outer_r_gen,((0.0 ,0.5, upper_loc)))
        lower_neighbour = rg.sphere(shape,beta*self.outer_r_gen,((1.0, 0.5, 1-upper_loc)))
        #neighbour = rg.sphere(shape,beta*self.outer_r_gen,((random.choice([0.0,1.0],1)[0], 0.5, random.choice([0.0,1.0],1)[0])))
        self.upper_neighbour = upper_neighbour
        self.lower_neighbour = lower_neighbour


        # add nucleoid as specified
        if self.nucleoid_type=='spherocylinder':
            nucld_cylinder = rg.cylinder(
                    shape = shape,
                    height = self.nucld_l_gen,
                    radius = self.nucld_r_gen,
                    axis = 0,
                    position = (0.5,0.5,0.5),
                    smoothing = False)
            nucld_sphere1 = rg.sphere(shape,self.nucld_r_gen,(0.5-0.5*self.nucld_l_gen/shape[0],0.5,0.5))
            nucld_sphere2 = rg.sphere(shape,self.nucld_r_gen,(0.5+0.5*self.nucld_l_gen/shape[0],0.5,0.5))
            self.nucleoid = nucld_cylinder+nucld_sphere1+nucld_sphere2
        
        elif self.nucleoid_type=='twin-spherocylinder':
            semi_nucld_l_gen = self.nucld_l_gen/2-self.nucld_r_gen

            #max_gap = self.nucld_r_gen#self.l_gen+self.inner_r_gen-self.nucld_l_gen-2*self.nucld_r_gen
            if self.with_IDP == True:
                max_gap = int( 0.5*self.l_gen + self.inner_r_gen - 2*IDP_r_gen - semi_nucld_l_gen - self.nucld_r_gen)
            else:
                max_gap = int((0.5*self.l_gen + self.inner_r_gen) - (semi_nucld_l_gen+self.nucld_r_gen))

            max_gap = max([max_gap,0])
            gap = random.randint(0,max_gap) #this gap is the distance between the center of the semi nucld sphere closer to the center of cell, and the center of cell
            
            semi_nucld_cylinder1 = rg.cylinder(
                    shape = shape,
                    height = semi_nucld_l_gen,
                    radius = self.nucld_r_gen,
                    axis = 0,
                    position = (0.5-(gap+0.5*semi_nucld_l_gen)/shape[0],0.5,0.5),
                    smoothing = False)
            semi_nucld_cylinder2 = rg.cylinder(
                    shape = shape,
                    height = semi_nucld_l_gen,
                    radius = self.nucld_r_gen,
                    axis = 0,
                    position = (0.5+(gap+0.5*semi_nucld_l_gen)/shape[0],0.5,0.5),
                    smoothing = False)
            semi_nucld1_sphere1 = rg.sphere(shape,self.nucld_r_gen,(0.5-(gap+semi_nucld_l_gen)/shape[0],0.5,0.5)) # for the upper semi nucleoid
            semi_nucld1_sphere2 = rg.sphere(shape,self.nucld_r_gen,(0.5-gap/shape[0],0.5,0.5))
            semi_nucld2_sphere1 = rg.sphere(shape,self.nucld_r_gen,(0.5+gap/shape[0],0.5,0.5)) # for the lower semi nucleoid
            semi_nucld2_sphere2 = rg.sphere(shape,self.nucld_r_gen,(0.5+(gap+semi_nucld_l_gen)/shape[0],0.5,0.5))
            self.nucleoid = semi_nucld_cylinder1+semi_nucld1_sphere1+semi_nucld1_sphere2+semi_nucld_cylinder2+semi_nucld2_sphere1+semi_nucld2_sphere2

        return self

    def sample_from_component(self, component, degree): 
        molecules_in_3D = np.zeros((component.shape[0], component.shape[1], component.shape[2]))
        valid_locations = np.argwhere(component==True)
        total_valid = len(valid_locations)
        numOfSamples = int(round(total_valid*degree))
        sampled_indices = np.random.choice(total_valid,numOfSamples,replace=True)
        for indx in sampled_indices:
            loc_indx = valid_locations[indx]
            molecules_in_3D[loc_indx[0], loc_indx[1], loc_indx[2]] += 30 #np.random.poisson(30)
        return molecules_in_3D
    
    def get_max_rotation_angle(self):
        if self.trench_width-2*self.outer_r > self.l:
            return 180
        else:
            return np.arcsin((self.trench_width-2*self.outer_r)/self.l)*180/np.pi

    def get_greyScale_image(self, objects, intensities, cutoff_options, rotation, centerloc): 
        '''returns a normalized image'''
        
        img = np.zeros((self.image_size_gen[0], self.image_size_gen[1]))
        photon_maps = np.zeros((objects[0].shape[0], objects[0].shape[1], len(objects)) )
        for i, object3D in enumerate(objects):
            projected = np.sum(object3D,axis=1)
            maxVal = np.max(projected)
            if cutoff_options[i]==1:
                projected[projected>self.brightness_cutoff*maxVal] = self.brightness_cutoff*maxVal
            photon_maps[:,:,i] = projected*intensities[i]/np.max(projected)

        object_img = np.sum(photon_maps, axis=2)
        #object_img = object_img/np.max(object_img)
        #object_img = (255*object_img).astype('uint8')

        centerloc = np.array(centerloc)
        img = place_img_on_background(object_img, img, centerloc)
        # topleftloc = np.round(np.multiply(centerloc,self.image_size_gen)-0.5*np.array(object_img.shape)).astype(int)
        # img[topleftloc[0]:topleftloc[0]+img_height,topleftloc[1]:topleftloc[1]+img_width] = object_img
        
        # rotate objects
        PIL_img = Image.fromarray(img)
        rotated_PIL_img = PIL_img.rotate(rotation)
        rotated_img = np.asarray(rotated_PIL_img)
        return rotated_img #type=float
