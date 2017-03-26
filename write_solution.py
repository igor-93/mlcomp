from skimage.draw import set_color
from skimage.draw import polygon_perimeter
from skimage.io import imsave
import skimage.color 


def write_output_classification(Y, f_names, name = "classifcation.txt"):
	f = open("output/" + name,'w') 
	for idx,(y,f_name) in enumerate(zip(Y,f_names)):
		f.write(f_name + ", " + str(y) + "\n")
	f.close()


##[upper_left_x,upper_left_y,width,height] <- one box
def write_output_detection(img,bounding_boxes):
	for idx in range(len(img)):
		image = skimage.color.hsv2rgb(img[idx])
		boxes = bounding_boxes[idx]
		for upper_left_x,upper_left_y,width,height in boxes:
			r = [upper_left_x,upper_left_x+width,upper_left_x+width,upper_left_x]
			c = [upper_left_y,upper_left_y,upper_left_y+height,upper_left_y+height]
			rr, cc = polygon_perimeter(r,c,image.shape,clip=True)
			image[rr,cc] = [0.0,128.0/256.0,0.0]
		imsave("output/images/pic" + str(idx+1) + ".jpg",image)
	
		
			
			
		
		
