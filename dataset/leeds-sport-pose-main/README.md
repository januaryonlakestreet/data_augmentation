# Leeds Sports Pose Extended Training Dataset

**Sam Johnson and Mark Everingham** : http://sam.johnson.io/research/lspet.html

This is a set of 10,000 images gathered from Flickr searches for the tags 'parkour', 'gymnastics', and 'athletics'. Each image has
a corresponding annotation gathered from Amazon Mechanical Turk.
The images have been scaled such that the annotated person is roughly 150 pixels in length.

The archive contains two top-level files and one folder:
* README.md - this document
* joints.mat - a MATLAB format matrix 'joints' consisting of 14 joint locations and visibility flags. Joints are labelled in the following order:
	* Right ankle
	* Right knee
	* Right hip
	* Left hip
	* Left knee
	* Left ankle
	* Right wrist
	* Right elbow
	* Right shoulder
	* Left shoulder
	* Left elbow
	* Left wrist
	* Neck
	* Head top
* images/ - 10,000 images


There is a second archive: http://sam.johnson.io/research/lspet_dataset_visualized.zip <br>
containing the 10,000 images above with rendered poses.


If you use this dataset please cite :

Sam Johnson and Mark Everingham <br>
"Learning Effective Human Pose Estimation from Inaccurate Annotation" <br>
In proceedings of Computer Vision and Pattern Recognition (CVPR) 2011

@inproceedings{Johnson11, <br>
	title = {Learning Effective Human Pose Estimation from Inaccurate Annotation}, <br>
	author = {Johnson, Sam and Everingham, Mark}, <br>
	year = {2011}, <br>
	booktitle = {Proceedings of Computer Vision and Pattern Recognition (CVPR) 2011} <br>
}

E-mail: s.a.johnson04@leeds.ac.uk
