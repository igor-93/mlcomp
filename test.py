import numpy as np
# x is horisontal
# y is vertical
def do_overlap(ax,ay,aw,ah,bx,by,bw,bh, cert_a, cert_b):
	box_area = min(aw*ah,bw*bh)

	dx = min(ax+aw, bx+bw) - max(ax, bx)
	dy = min(ay+ah, by+bh) - max(ay, by)

	overlaped_area = dx*dy

	overlap_ratio = overlaped_area / box_area

	if overlap_ratio >= 0.5:
		if cert_a > cert_b:
			return 1
		else:
			return 2
	else:
		return 0 

# box: x,y,width, height
# x horizontal
def remove_overlapping(boxes, certainties):
	found_overlap = False
	remaining_boxes = []
	remaining_cert = []


	i = 0
	while True:
		if i >= len(boxes)-1:
			break

		box1 = boxes[i]
		box2 = boxes[i+1]
		res = do_overlap(box1[0], box1[1], box1[2], box1[3], box2[0], box2[1], box2[2], box2[3], certainties[i], certainties[i+1])

		if res == 0:
			remaining_boxes.append(box1)
			remaining_cert.append(certainties[i])
			#remaining_boxes.append(box2)
			i += 1
		if res == 1:
			remaining_boxes.append(box1)
			remaining_cert.append(certainties[i])
			found_overlap = True
			i += 2
		if res == 2:
			remaining_boxes.append(box2)
			remaining_cert.append(certainties[i+1])	
			found_overlap = True
			i += 2

	return remaining_boxes, remaining_cert, found_overlap


# box: x,y,width, height
# x horizontal
def main_remove_overlapping(boxes, cert):
	keep_box = len(boxes) * [True]

	for i in range(len(boxes)):
		for j in range(i):

			box1 = boxes[i]
			box2 = boxes[j]
			if keep_box[i] == False or keep_box[j] == False:
				continue
			res = do_overlap(box1[0], box1[1], box1[2], box1[3], box2[0], box2[1], box2[2], box2[3], cert[i], cert[j])
			if res == 0:
				#keep_box[i] = True
				#keep_box[j] = True
				pass
			if res == 1:
				#eep_box[i] = True
				keep_box[j] = False
			if res == 2:
				keep_box[i] = False
				#keep_box[j] = True

	boxes = np.array(boxes)
	return boxes[keep_box]


