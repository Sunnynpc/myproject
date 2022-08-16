import glob
import xml.etree.ElementTree as ET

import numpy as np

from kmeans import kmeans, avg_iou

ANNOTATIONS_PATH = "F:\YOLOV3\label\labels\labels.txt"
CLUSTERS = 9


def load_dataset(path):
    dataset = []
    with open(path) as f:
        dataset_1 = f.readlines()
    for i in range(10):
        line = dataset_1[i]
        strs = line.split()
        _boxes = np.array([float(x) for x in strs[1:]])
        boxes = np.split(_boxes, len(_boxes) // 5)
        boxes = np.stack(boxes)
        box1 = boxes[:,3:]
        for i,x in enumerate(box1):
            dataset.append(box1[i,:])
    dataset = np.array(dataset)

    # for xml_file in glob.glob("{}/*xml".format(path)):
    #     tree = ET.parse(xml_file)
    #
    #     height = int(tree.findtext("./size/height"))
    #     width = int(tree.findtext("./size/width"))
    #
    #     try:
    #         for obj in tree.iter("object"):
    #             xmin = int(obj.findtext("bndbox/xmin")) / width
    #             ymin = int(obj.findtext("bndbox/ymin")) / height
    #             xmax = int(obj.findtext("bndbox/xmax")) / width
    #             ymax = int(obj.findtext("bndbox/ymax")) / height
    #
    #             xmin = np.float64(xmin)
    #             ymin = np.float64(ymin)
    #             xmax = np.float64(xmax)
    #             ymax = np.float64(ymax)
    #             if xmax == xmin or ymax == ymin:
    #                 print(xml_file)
    #             dataset.append([xmax - xmin, ymax - ymin])
    #     except:
    #         print(xml_file)
    return dataset


if __name__ == '__main__':
    # print(__file__)
    data = load_dataset(ANNOTATIONS_PATH)
    # print(data)
    # print(data.shape)
    out = kmeans(data, k=CLUSTERS)
    # clusters = [[10,13],[16,30],[33,23],[30,61],[62,45],[59,119],[116,90],[156,198],[373,326]]
    # out= np.array(clusters)/416.0
    print(out)
    print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
    print("Boxes:\n {}-{}".format(out[:, 0] , out[:, 1] ))

    ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
    print("Ratios:\n {}".format(sorted(ratios)))

#62*150=9300  182*114=20748 81.5*292.5=23652 202*77=15939