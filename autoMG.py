import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import cv2
import sys

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from pathlib import Path



torch.cuda.is_available = lambda : False
#srcPath = Path( F"D:/Projects/SoilLifeConsultant/SAMtest1" )   # pathInput  = "C:\\Users\\Owner\\smoIDuserO\\train\\"
#srcPath = "D:\\Projects\\SoilLifeConsultant\\SAMtest1"
srcPath = "./SAMtest1/"
outPath = "./Output/"
using_colab = False
#sys.path.append("..")
#-------------------------------------------------------------------------------------------


#-------------------------------------------------------------------------------------------
def log( txt, outPutFileName=None ):

    # 10/29/21: need to add numpy arrays:  ( "ndarray" in str( type( txt ) ) )
    if ( outPutFileName == None ):
        outPutFileName = "logFile.txt"
    
    with open( outPutFileName, "a" ) as lF:
        
        if ( "list" in str( type( txt ) ) ):
           rng  = range(0,len( txt ))
           for i in rng:
               lF.write( str( txt[i] ) )
               lF.write( "\n" )

        elif ( "dict" in str( type( txt ) ) ):
           for key in txt:
               lF.write( key + ": " + str( txt[key] ) )
               lF.write( "\n" )

        elif ( "str" in str( type( txt ) ) ):
            lF.write( txt )
            lF.write( "\n" )

        elif ( "bool" in str( type( txt ) ) ):
            if ( txt ):
                lF.write( "True" )
            else:
                lF.write( "False" )

        elif ( "int" in str( type( txt ) ) ) or ( "float" in str( type( txt ) ) ):
            lF.write( str( txt ) )
            lF.write( "\n" )

        elif ( "ndarray" in str( type( txt ) ) ):
            for row in txt:
                np.savetxt( lF, row )

    lF.close
    return
#-------------------------------------------------------------------------------------------


#-------------------------------------------------------------------------------------------
# def showAnns( anns ):

#     if len(anns) == 0:
#         return

#     sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
#     ax = plt.gca()
#     ax.set_autoscale_on(False)

#     img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
#     img[:,:,3] = 0
#     for ann in sorted_anns:
#         m = ann['segmentation']
#         color_mask = np.concatenate([np.random.random(3), [0.35]])
#         img[m] = color_mask
#     ax.imshow(img)
#     #cv2.imwrite( "test.png", img )
#     #plt.savefig( img )       # "test.png", 
#     return img

def showAnns( anns, save_path=None ):

    if len(anns) == 0:
        return

    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)
    
    if save_path is not None:
        plt.imsave(save_path, img)
    return img

#-------------------------------------------------------------------------------------------


#-------------------------------------------------------------------------------------------
def main(): 

    # Example image
    #image = cv2.imread( "D:/Projects/SoilLifeConsultant/SAMtest1/malena4_Alaimus_1.jpg" )
    #image = cv2.cvtColor( image, cv2.COLOR_BGR2RGB )
    #plt.figure(figsize=(20,20))
    #plt.imshow(image)
    #plt.axis('off')
    #plt.show()

    pixList = [srcPath+file for file in os.listdir( srcPath ) if file.endswith(('png', 'jpg'))]

    for theP in pixList:

        image = cv2.imread( theP )
        image = cv2.cvtColor( image, cv2.COLOR_BGR2RGB )

        # Automatic mask generation
        # To run automatic mask generation, provide a SAM model to the `SamAutomaticMaskGenerator` class. Set the path below to the SAM checkpoint. Running on CUDA and with the default model is recommended.

        sam_checkpoint = "/tank0/ai_models/segment_anything/sam_vit_h_4b8939.pth"
        model_type        = "vit_h"
        device                = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )
        sam                    = sam_model_registry[model_type](checkpoint=sam_checkpoint)

        sam.to( device=device )
        mask_generator = SamAutomaticMaskGenerator( sam )

        # To generate masks, just run "generate" on an image.
        masks = mask_generator.generate( image )

        # Mask generation returns a list over masks, where each mask is a dictionary containing various data about the mask. These keys are:
        # * `segmentation` : the mask
        # * `area` : the area of the mask in pixels
        # * `bbox` : the boundary box of the mask in XYWH format
        # * `predicted_iou` : the model's own prediction for the quality of the mask
        # * `point_coords` : the sampled input point that generated this mask
        # * `stability_score` : an additional measure of mask quality
        # * `crop_box` : the crop of the image used to generate this mask in XYWH format

        print( "Number masks, v.1: ", len(masks) )
        print( "Mask v.1 keys: ", masks[0].keys() )

        # Show all the masks overlayed on the image.
        plt.figure( figsize=(20,20) )
        plt.imshow( image )
        maskedImg = showAnns( masks )
        #cv2.imwrite( theP + ".png", maskedImg )
        plt.axis( "off" )
        plt.show() 
        plt.savefig( "test.png" )
        ##mImg = cv2.imfuse( image, maskedImg, "blend", "Scaling", "joint" )
        ##cv2.imwrite( theP + ".png", mImg )

        # Automatic mask generation options
        # There are several tunable parameters in automatic mask generation that control how densely points are sampled and what the thresholds are for removing low quality or duplicate masks. 
        #Additionally, generation can be automatically run on crops of the image to get improved performance on smaller objects, and post-processing can remove stray pixels and holes. 
        #Here is an example configuration that samples more masks:

        mask_generator_2 = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,  # Requires open-cv to run post-processing
        )

        masks2 = mask_generator_2.generate( image )
        print( "Number masks, v.1: ", len(masks2) )

        plt.figure( figsize=(20,20) )
        plt.imshow( image )
        showAnns( masks2 , "test2.png")
        plt.axis('off')
        plt.show() 
        plt.close( "all" )
#-------------------------------------------------------------------------------------------


#-------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
#-------------------------------------------------------------------------------------------

