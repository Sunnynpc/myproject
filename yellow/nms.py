import numpy as np

def iou(box,boxes,is_min=False):
    # 计算矩形框的面积
    box_area = (box[2]-box[0])*(box[3]-box[1])
    boxes_area = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])

    # 交集
    x1 = np.maximum(box[0],boxes[:,0])
    y1 = np.maximum(box[1],boxes[:,1])
    x2 = np.minimum(box[2],boxes[:,2])
    y2 = np.minimum(box[3],boxes[:,3])

    # 判断是否有交集
    w = np.maximum(0,x2-x1)
    h = np.maximum(0,y2-y1)

    # 求交集的面积
    inter = w*h
    if is_min:
        ovr = np.true_divide(inter,np.minimum(box_area,boxes_area))
    else:
        ovr = np.true_divide(inter,(box_area+boxes_area-inter))
    return ovr

def nms(boxes,thresh=0.3,is_min=False):
    if boxes.shape[0] == 0:
        return np.array([])
    # 按置信度由大到小进行排序
    _boxes = boxes[(-boxes[:,4]).argsort()]
    # 创建空的列表存放保留下来的框
    r_boxes = []
    while _boxes.shape[0]>1:
        # 拿出第一个框
        box1 = _boxes[0]
        # 拿出剩下的框
        boxes1 = _boxes[1:]
        # 将第一个框保存
        r_boxes.append(box1)
        # 计算iou，将小于阈值的框保留
        index = np.where(iou(box1,boxes1,is_min=False)<thresh)
        _boxes = boxes1[index]
    # 保留最后一个框
    if _boxes.shape[0] > 0:
        r_boxes.append(_boxes[0])
    return np.stack(r_boxes)