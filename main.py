import matplotlib.pyplot as plt
import cv2 as cv



import data_backend_operations as db
import preprocessing as pr
import segmentation as seg
import classification as cls

def main(single_image=None, show_step_pictures=False, show_examples=False):
    classifier_data  = db.import_vector_for_cls()
    if classifier_data is not None:
        for i in range(0, 15):
            base_img = db.get_img_from_dataset(i)
            img_ = db.resize_picture(base_img)
            if show_step_pictures:
                db.print_img(img_, title="Image after resizing")
            img = pr.convert_BGR2HSV(img_)
            img_tresholded = pr.get_treshold(img, 22, 8)
            if show_step_pictures:
                db.print_img(img_tresholded, title="Image after tresholding")
            img_closed = pr.make_binary_operations(img_tresholded)
            if show_step_pictures:
                db.print_img(img_closed, title="Image after closing")
                plt.imshow(img_closed)
                plt.show()
            img_segmented, seg_no, segments = seg.get_segments(img_closed)
            print(f"Possible segments found: {seg_no}")
            img__ = img_
            for segment in segments:
                point_min, point_max = seg.determine_extreme_points_seg(segment["cordinates"])
                img__ = cv.rectangle(img__, point_min, point_max, color=(255, 0, 0), thickness=2)
                # TODO visualise bbox function
                segm_bbox = seg.crop_segment(img_segmented, segment["cordinates"])
                segm_bbox_border, perimeter, _ = seg.get_Moore_Neighborhood_countour(segm_bbox, segment["key"])
                img_segmented[point_min[1]:point_max[1], point_min[0]:point_max[0]] = segm_bbox
                features = cls.calculate_invariants(segment)
                area = cls.calculate_area(segment)
                features[0] = cls.calculate_Malinowska_ratio(area, perimeter)
                is_M_logo = cls.check_segment(features, classifier_data["feature value"], classifier_data["standard deviation"])
                if is_M_logo is True:
                    img_ = cv.rectangle(img_, point_min, point_max, color=(0, 255, 0), thickness=2)
            if show_step_pictures:
                db.print_img(img__, "Annotaded bboxes for all segments")
            db.print_img(img_, "Founded logos")
    else:
        print("No data for clasification uploaded")
        return 0

if __name__ == "__main__":
    main(show_step_pictures=True)
