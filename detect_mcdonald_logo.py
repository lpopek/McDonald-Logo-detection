#!/usr/bin/python

import sys, getopt
import copy
import matplotlib.pyplot as plt
import cv2 as cv

import modules.data_backend_operations as db
import modules.preprocessing as pr
import modules.segmentation as seg
import modules.classification as cls

def detect_logo(base_img,  show_step_pictures=False, save=False):
    classifier_data  = db.import_vector_for_cls()
    if classifier_data is not None:
        img_ = db.resize_picture(base_img)
        img_rgb = pr.convert_BGR2RGB(img_)
        if show_step_pictures:
            db.print_img(img_rgb, title="Image after resizing")
        img = pr.convert_BGR2HSV(img_)
        img_tresholded = pr.get_treshold(img, 22, 8)
        if show_step_pictures:
            db.print_img(img_tresholded, title="Image after tresholding", gray_scale_flag=True)
        img_closed = pr.make_binary_operations(img_tresholded)
        if show_step_pictures:
            db.print_img(img_closed, title="Image after closing", gray_scale_flag=True)
        img_segmented, seg_no, segments = seg.get_segments(img_closed)
        
        img__ = copy.deepcopy(img_rgb)
        save_flag = False
        for segment in segments:
            point_min, point_max = seg.determine_extreme_points_seg(segment["cordinates"])
            img__ = db.create_bbox(img__, point_min, point_max, color=(255, 0, 0), thickness=2)
            segm_bbox = seg.crop_segment(img_segmented, segment["cordinates"])
            segm_bbox_border, perimeter, _ = seg.get_Moore_Neighborhood_countour(segm_bbox, segment["key"])
            img_segmented[point_min[1]:point_max[1], point_min[0]:point_max[0]] = segm_bbox
            features = cls.calculate_invariants(segment)
            area = cls.calculate_area(segment)
            features[0] = cls.calculate_Malinowska_ratio(area, perimeter)
            is_M_logo = cls.check_segment(features, classifier_data["feature value"], classifier_data["standard deviation"])
            if is_M_logo is True:
                save_flag = True
                img_rgb = db.create_bbox(img_rgb, point_min, point_max, color=(0, 255, 0), thickness=2)
        print(f"possible segments found: {seg_no}")
        if show_step_pictures:
            db.print_img(img__, "Annotaded bboxes for all segments")
        db.print_img(img_rgb, "Founded logos")
        if save:
            if save_flag:
                plt.imsave("save\detection_result.png", img_rgb)
                return 0
            else:
                print("No detected logos. Save skipped")
                return -1
    else:
        print("No data for clasification uploaded")
        return -1
    
def main(argv):
    argumentList = sys.argv[1:]

    options = "hinds"
    long_options = ["Help", "Image", "Dataset_number", ]
    display_steps = False
    img = None
    n  = -1
    img_loaded = False
    save_flag = False
    try:
        # Parsing argument
        arguments, values = getopt.getopt(argumentList, options, long_options)
        
        # checking each argument
        for currentArgument, currentValue in arguments:
            if currentArgument in ("-h", "--Help"):
                print ("-i <inputimage> \n\
                        -n <number_from_ds> - make detection from exaples \n\
                        -d <bool> - display steps during detection \n\
                        -h -display help \n\
                        -s -save detection result to .png\n")
                sys.exit()
            elif currentArgument in ("-i", "--Image"):
                print ("Loading image: ", sys.argv[1])
                img = cv.imread(currentValue)
                img_loaded = True
            elif currentArgument in ("-n", "--Dataset_number"):
                print ("Loading image from dataset number: ", sys.argv[1])
                n = currentValue
                if n > 15 and n < -1:
                    print("wrong number")
                    sys.exit()
            elif currentArgument in ("-d", "--Display"):
                display_steps = True
            elif currentArgument in ("-s", "--Save"):
                save_flag = True
        if img_loaded:
            detect_logo(img, show_step_pictures=display_steps, save=save_flag)
            sys.exit()
        img = db.get_img_from_dataset(n)
        if img is not None:
            detect_logo(img, show_step_pictures=display_steps, save=save_flag)
            sys.exit()
        else:
            print("No image loaded")

    
    except getopt.error as err:
        print (str(err))


if __name__ == "__main__":
    main(sys.argv[1:])