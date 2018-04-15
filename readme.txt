Setup:
	Dependencies:
		Python2.7
		Keras: https://github.com/keras-team/keras
		TensorFlow: https://www.tensorflow.org/
		Scipy: https://www.scipy.org/
		Numpy: http://www.numpy.org/
Usage:
	Add content image
	Open style.py and add the content image path into 'base_image_path'
	
	Add style image
	Open style.py and you can add one style image or more than one images path into 'style_img'	

	Basic Use
	Run 'python style.py'
	it will generate a 400X400 combination style image, because small size image cost less time, if you want change the size of image, you can update img_nrows and img_ncols

	Calculate style loss 
	Input 1 content image and 1 style image, Set 'show_style_loss' to 1, and run 'python style.py' the program will calculate the style loss between the content image and style image 		you input.

	Calculate content loss 
	Input 1 content image and 1 style image, Set 'show_content_loss' to 1, and run 'python style.py' the program will calculate the style loss between the content image and style image 		you input.
