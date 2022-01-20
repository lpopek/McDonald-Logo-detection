import re
import cv2 as cv



import data_backend_operations as db
import preprocessing as pr
import segmentation as seg
import classification as cls

def main():
    classifier_data  = db.import_vector_for_cls()
    if classifier_data is not None:
        for i in range(0, 15):
            base_img = db.get_img_from_dataset(i)
            img_ = db.resize_picture(base_img)
            # db.print_img(img_, title="Obraz po wyjściowy")
            img = pr.convert_BGR2HSV(img_)
            img_tresholded = pr.get_treshold(img, 22, 8)
            # db.print_img(img_tresholded, title="Obraz po progowaniu", gray_scale_flag=True)
            img = pr.make_binary_operations(img_tresholded)
            # db.print_img(img, title="Obraz po zamknieciu", gray_scale_flag=True)
            img_segmented, seg_no, segments = seg.get_segments(img)
            print(f"Possible segments found: {seg_no}")
            # db.print_img(img_segmented, title="Obraz po segemntacji")
            i = 0
            for segment in segments:
                point_min, point_max = seg.determine_extreme_points_seg(segment["cordinates"])
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
            db.print_img(img_, "Zaznaczone bboxy")
    else:
        print("No data for clasification uploaded")
        return 0

if __name__ == "__main__":
    main()
