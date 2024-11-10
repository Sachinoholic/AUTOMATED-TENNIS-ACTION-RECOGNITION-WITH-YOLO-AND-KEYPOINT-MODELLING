import math
def get_center_of_bbox(bbox):
    x1,y1,x2,y2  = bbox
    center_x = int((x1+x2)/2)
    center_y = int((y1+y2)/2)
    return (center_x,center_y)
def measure_distance(p1,p2):
    return ((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)**5
