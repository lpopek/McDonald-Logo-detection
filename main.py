import cv2 as cv



import data_backend_operations as db
import preprocessing as pr
import segmentation as seg

def main():
    IMG = cv.imread("test\test.png")


if __name__ == "__main__":
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
        # db.print_img(img_segmented, title="Obraz po segemntacji")
        i = 0
        print("ready for segment annotation")
        for segment in segments:
            point_min, point_max = seg.determine_extreme_points_seg(segment["cordinates"])
            # TODO visualise bbox function
            img_ = cv.rectangle(img_, point_min, point_max, color=(255, 0, 0), thickness=2)
            p1, p2 = seg.determine_extreme_points_seg(segment["cordinates"])
            segm_bbox = seg.crop_segment(img_segmented, segment["cordinates"])
            segm_bbox, perimeter, _ = seg.get_Moore_Neighborhood_countour(segm_bbox, segment["key"])
            img_segmented[p1[1]:p2[1], p1[0]:p2[0]] = segm_bbox

        db.print_img(img_, "Zaznaczone bboxy")
