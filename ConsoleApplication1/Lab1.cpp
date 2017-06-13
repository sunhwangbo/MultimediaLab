#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <vector>
using namespace cv;
using namespace std;

/*
int main(int argc, char** argv)
{
	Mat color_img(Size(1920, 1080), CV_8UC3);
	color_img = imread("C:\\Program Files\\opencv\\sources\\samples\\data\\lena.jpg", IMREAD_COLOR); // Read the file
	Mat result_img = color_img.clone();
	if (color_img.empty()) // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	for (int x = 0; x < color_img.rows; x++)
	{
		for (int y = 0; y < color_img.cols; y++)
		{
			result_img.at<Vec3b>(y, x)[0] = saturate_cast<uchar>(color_img.at<Vec3b>(y, x)[0] * 2.2 + 50);
			result_img.at<Vec3b>(y, x)[1] = saturate_cast<uchar>(color_img.at<Vec3b>(y, x)[1] * 2.2 + 50);
			result_img.at<Vec3b>(y, x)[2] = saturate_cast<uchar>(color_img.at<Vec3b>(y, x)[2] * 2.2 + 50);
		}
	}
	namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Display window", result_img); // Show our image inside it.

	waitKey(0); // Wait for a keystroke in the window
	return 0;
}*/

/*int main(int argc, char** argv)
{
	float a = 0.7, b = 0.3;
	Mat image1, image2;
	image1 = imread("C:\\Program Files\\opencv\\sources\\samples\\data\\lena.jpg", IMREAD_COLOR); // Read the file
	image2 = imread("C:\\Program Files\\opencv\\sources\\samples\\data\\LinuxLogo.jpg", IMREAD_COLOR); // Read the file

	cv::resize(image1, image1, image2.size(), 0, 0, CV_INTER_LINEAR);

	Mat dst = Mat::zeros(image2.size(), image2.type());

	if (image1.empty() && image2.empty()) // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	for (int y = 0; y < dst.rows; y++)
	{
		for (int x = 0; x < dst.cols; x++)
		{
			dst.at<Vec3b>(y, x)[0] = saturate_cast<uchar>(image1.at<Vec3b>(y, x)[0] * a + image2.at<Vec3b>(y, x)[0] * b);
			dst.at<Vec3b>(y, x)[1] = saturate_cast<uchar>(image1.at<Vec3b>(y, x)[1] * a + image2.at<Vec3b>(y, x)[1] * b);
			dst.at<Vec3b>(y, x)[2] = saturate_cast<uchar>(image1.at<Vec3b>(y, x)[2] * a + image2.at<Vec3b>(y, x)[2] * b);
		}
	}

	namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
	if (!dst.empty())
		imshow("Display window", dst); // Show our image inside it.
	else
		printf("empty");

	waitKey(0); // Wait for a keystroke in the window
	return 0;
}*/

/*
int main(int argc, char** argv)
{
	Mat src = imread("D:\\Lecture\\4학년\\멀티미디어\\helmet.02.fadjust0.0.bmp", IMREAD_COLOR); // Read the file
	Mat dst;
	if (src.empty()) // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	cvtColor(src, src, CV_BGR2GRAY);
	equalizeHist(src, dst);
	namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Display window", src); // Show our image inside it.

	namedWindow("Display window2", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Display window2", dst); // Show our image inside it.
	waitKey(0); // Wait for a keystroke in the window
	return 0;
}
*/

/*
int smoothing(int i, int j, int size);
Mat src, dst;
int main(int argc, char** argv)
{
	src = imread("C:\\Program Files\\opencv\\sources\\samples\\data\\lena.jpg", IMREAD_COLOR); // Read the file
	int size;

	if (src.empty()) // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	cvtColor(src, src, CV_BGR2GRAY);
	printf("enter filter size: ");
	scanf("%d", &size);

	copyMakeBorder(src, src, size, size, size, size, 0, 0);

	dst = src.clone();
	for (int i = size; i < src.rows-size; i++)
	{
		for (int j = size; j < src.cols -size; j++)
		{
			dst.at<uchar>(i,j) = smoothing(i, j, size) / pow(size*2+1,2);
		}
	}
	namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Display window", src); // Show our image inside it.

	namedWindow("Display window2", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Display window2", dst); // Show our image inside it.
	waitKey(0); // Wait for a keystroke in the window
	return 0;
}

int smoothing(int i, int j , int size) {
	int temp = 0;
	for (int k = -size ; k <= size; k++)
	{
		for (int r = -size; r <= size; r++)
			temp += src.at<uchar>(i + k, j + r);
	}

	return temp;
}*/


/*
uchar median_filter(int i, int j, int size);
int ucharcmp(int a, int b);
Mat src, dst;
int* mask;
int mask_size;
int main(int argc, char** argv)
{
	src = imread("D:\\Lecture\\4학년\\멀티미디어\\lena_noise.png", IMREAD_COLOR); // Read the file
	int size;

	if (src.empty()) // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	cvtColor(src, src, CV_BGR2GRAY);
	printf("enter filter size: ");
	scanf("%d", &size);
	mask_size = pow(size * 2 + 1, 2);
	dst = src.clone();
	copyMakeBorder(src, src, size, size, size, size, 0, 0);
	mask = (int*)malloc(sizeof(int)*mask_size);
	for (int i = size; i < dst.rows - size; i++)
	{
		for (int j = size; j < dst.cols - size; j++)
		{
			dst.at<uchar>(i, j) = median_filter(i, j, size);
		}
	}
	free(mask);
	namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Display window", src); // Show our image inside it.

	namedWindow("Display window2", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Display window2", dst); // Show our image inside it.
	waitKey(0); // Wait for a keystroke in the window
	return 0;
}

int ucharcmp(const void* a, const void* b) {
	return *(int*)a - *(int*)b;
}
uchar median_filter(int i, int j, int size) {
	int index = 0;
	for (int k = -size; k <= size; k++)
	{
		for (int r = -size; r <= size; r++)
			mask[index++] = src.at<uchar>(i + k, j + r);
	}
	qsort(mask, mask_size, sizeof(int), ucharcmp);
	return mask[mask_size/2];
}*/

/*
uchar sobel_filter(int i, int j);
Mat src, dst;
int A[3][3] = { 1,0,-1,2,0,-2,1,0,-1 };
int B[3][3] = { 1,2,1,0,0,0,-1,-2,-1 };

int main(int argc, char** argv)
{
	src = imread("C:\\Program Files\\opencv\\sources\\samples\\data\\lena.jpg", IMREAD_COLOR); // Read the file
	int size = 1;

	if (src.empty()) // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	cvtColor(src, src, CV_BGR2GRAY);
	dst = src.clone();
	copyMakeBorder(src, src, size, size, size, size, 0, 0);
	for (int i = size; i < dst.rows - size; i++)
	{
		for (int j = size; j < dst.cols - size; j++)
		{
			dst.at<uchar>(i, j) = sobel_filter(i, j);
		}
	}

	namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Display window", src); // Show our image inside it.

	namedWindow("Display window2", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Display window2", dst); // Show our image inside it.
	waitKey(0); // Wait for a keystroke in the window
	return 0;
}

uchar sobel_filter(int i, int j) {
	int x = 0, y = 0;
	for (int k =0; k < 3; k++)
	{
		for (int r = 0; r < 3; r++) {
			x += A[k][r] * src.at<uchar>(i + (k - 1), j + (r - 1));
			y += B[k][r] * src.at<uchar>(i + (k - 1), j + (r - 1));
		}
	}

	return sqrt(pow(x, 2) + pow(y, 2));
}*/

/*
int smoothing(int i, int j, int size);
int laplacian(int i, int j);
int mask[3][3] = { {1,1,1},
{1,-8,1},
{1,1,1} };
int mask2[3][3] = { { 0,-1,0 },
{ -1,4,-1 },
{ 0,-1,0 } };
Mat src, dst, dst2, dst3;
int main(int argc, char** argv)
{
	src = imread("C:\\Program Files\\opencv\\sources\\samples\\data\\lena.jpg", IMREAD_COLOR); // Read the file
	int size = 2;

	if (src.empty()) // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	cvtColor(src, src, CV_BGR2GRAY);
	copyMakeBorder(src, src, size, size, size, size, 0, 0);
	dst = src.clone();
	dst2 = src.clone();
	dst3 = src.clone();
	for (int i = size; i < src.rows - size; i++)
	{
		for (int j = size; j < src.cols - size; j++)
		{
			dst.at<uchar>(i, j) = smoothing(i, j, size) / pow(size * 2 + 1, 2);
		}
	}

	for (int i = size; i < src.rows - size; i++)
	{
		for (int j = size; j < src.cols - size; j++)
		{
			dst2.at<uchar>(i, j) = saturate_cast<uchar>(laplacian(i, j));
		}
	}
	for (int i = size; i < src.rows - size; i++)
	{
		for (int j = size; j < src.cols - size; j++)
		{
			dst3.at<uchar>(i, j) = saturate_cast<uchar>(dst.at<uchar>(i, j) + laplacian(i,j));
		}
	}
	namedWindow("Display window2", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Display window2", dst); // Show our image inside it.

	namedWindow("Display window3", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Display window3", dst2); // Show our image inside it.

	namedWindow("Display window4", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Display window4", dst3); // Show our image inside it.

	waitKey(0); // Wait for a keystroke in the window
	return 0;
}

int smoothing(int i, int j, int size) {
	int temp = 0;
	for (int k = -size; k <= size; k++)
	{
		for (int r = -size; r <= size; r++)
			temp += src.at<uchar>(i + k, j + r);
	}

	return temp;
}

int laplacian(int i, int j)
{
	int x = 0;
	for (int k = 0; k < 3; k++)
	{
		for (int r = 0; r < 3; r++) {
			x += mask2[k][r] * dst.at<uchar>(i + (k - 1), j + (r - 1));
		}
	}
	return x;
}*/

/*
Mat src, dst;
int main(int argc, char** argv)
{
	src = imread("C:\\Program Files\\opencv\\sources\\samples\\data\\lena.jpg", IMREAD_COLOR); // Read the file
	int size = 0;
	double r, s;
	double sigma = 0.0,sum = 0.0, temp = 0.0;
	double** kernel;
	int mask_length = 0;
	if (src.empty()) // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	printf("enter sigma\n");
	scanf("%lf", &sigma);
	printf("enter size\n");
	scanf("%d", &size);
	s = pow(sigma, 2)*2.0;
	mask_length = size * 2 + 1;

	cvtColor(src, src, CV_BGR2GRAY);
	dst = src.clone();
	copyMakeBorder(src, src, size, size, size, size, 0, 0);

	// allocate the memory for the kernel
	kernel = (double**)malloc(sizeof(double*)*mask_length);
	for (int i = 0; i< mask_length; i++)
		kernel[i] = (double*)malloc(sizeof(double)*mask_length);

	// generate the kernel
	for (int i = -size; i <= size; i++)
	{
		for (int j = -size; j <= size; j++)
		{
			r = sqrt(pow(i,2) + pow(j,2));
			kernel[i + size][j + size] = (exp(-(pow(r, 2) / s))) / (3.14*s);
			sum += kernel[i + size][j + size];
		}
	}
	// normalize the kernel
	for (int i = 0; i < mask_length; i++)
	{
		for (int j = 0; j < mask_length; j++)
		{
			kernel[i][j] /= sum;
		}
	}
	// convolution
	for (int i = size; i < dst.rows - size; i++)
	{
		for (int j = size; j < dst.cols - size; j++)
		{
			temp = 0.0;
			for (int k = -size; k <= size; k++)
			{
				for (int r = -size; r <= size; r++)
				{
					temp += kernel[k + size][r + size] * src.at<uchar>(i + k, j + r);
				}
			}
			dst.at<uchar>(i,j) = saturate_cast<double>(temp);
		}
	}

	namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Display window", src); // Show our image inside it.

	namedWindow("Display window2", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Display window2", dst); // Show our image inside it.
	waitKey(0); // Wait for a keystroke in the window

	free(kernel);
	return 0;
}
*/

/*
void convert_cmy(Mat,Mat);
int main(int argc, char** argv)
{
	Mat src = imread("C:\\Program Files\\opencv\\sources\\samples\\data\\baboon.jpg", IMREAD_COLOR); // Read the file
	float k = 0.7;
	if (src.empty()) // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	Mat hsv, ycbcr;
	Mat cmy(Size(src.cols,src.rows),CV_8UC3);
	cvtColor(src, hsv, CV_BGR2HSV);
	cvtColor(src, ycbcr, CV_BGR2YCrCb);
	convert_cmy(src,cmy);

	namedWindow("Original", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Original", src); // Show our image inside it.

	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			hsv.at<Vec3b>(i, j)[2] *= k;
			ycbcr.at<Vec3b>(i, j)[0] *= k;
			for (int r = 0; r < 3; r++)
			{
				cmy.at<Vec3b>(i, j)[r] = k*cmy.at<Vec3b>(i, j)[r] + (1 - k) * 255;
				src.at<Vec3b>(i, j)[r] = k*src.at<Vec3b>(i, j)[r];
			}
		}
	}
	convert_cmy(cmy, cmy);
	namedWindow("Original2", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Original2", src); // Show our image inside it.

	namedWindow("CMY", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("CMY", cmy); // Show our image inside it.

	cvtColor(hsv, hsv, CV_HSV2BGR);
	namedWindow("HSV", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("HSV", hsv); // Show our image inside it.

	cvtColor(ycbcr, ycbcr, CV_YCrCb2BGR);
	namedWindow("YCbCr", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("YCbCr", ycbcr); // Show our image inside it.

	waitKey(0); // Wait for a keystroke in the window

	return 0;
}
void convert_cmy(Mat src, Mat cmy) {
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			for (int k = 0; k < 3; k++)
				cmy.at<Vec3b>(i, j)[k] = 255 - src.at<Vec3b>(i, j)[k];
		}
	}
}*/

/*
int main()
{
	Mat src = imread("C:\\Program Files\\opencv\\sources\\samples\\data\\lena.jpg", IMREAD_GRAYSCALE); // Read the file
	float ratio = 0.0;
	scanf("%f", &ratio);

	float x, y;
	Mat dst(Size(src.rows * ratio, src.cols * ratio), CV_8UC1);
	ratio = 1 / ratio;
	for (int i = 0; i < dst.cols; i++)
	{
		for (int j = 0; j < dst.rows; j++)
		{
			x = floor(j*ratio);
			y = floor(i*ratio);
			dst.at<uchar>(j,i) = (int)src.at<uchar>(x, y);
		}
	}
	namedWindow("Original", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Original", src); // Show our image inside it.

	namedWindow("Resize", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Resize", dst); // Show our image inside it.
	waitKey(0); // Wait for a keystroke in the window

	return 0;
}*/

/*
int main()
{
	Mat src = imread("C:\\Program Files\\opencv\\sources\\samples\\data\\lena.jpg", IMREAD_GRAYSCALE); // Read the file
	float ratio = 0.0;
	scanf("%f", &ratio);

	float a,b,c,d;
	int p, q;
	Mat dst(Size(src.rows * ratio, src.cols * ratio), CV_8UC1);
	for (int y = 0; y < dst.rows; y++)
	{
		for (int x = 0; x < dst.cols; x++)
		{
			float px = floor(x / ratio);
			float py = floor(y / ratio);

			//calculate neighbor pixels ratio
			float fx = x / ratio - px;
			float fx2 = 1 - fx;
			float fy = y / ratio - py;
			float fy2 = 1 - fy;
			a = fx2 *fy2;
			b = fx * fy2;
			c = fx2 * fy;
			d = fx * fy;

			//calculate pixel value
			dst.at<uchar>(y, x) = a*src.at<uchar>(py, px)
				+ b*src.at<uchar>(py, px + 1 > src.cols-1 ? src.cols - 1 : px+1)
				+ c*src.at<uchar>(py + 1 > src.rows-1 ? src.rows - 1 : py+1, px)
				+ d*src.at<uchar>(py + 1>  src.rows - 1 ? src.rows - 1 : py + 1, px + 1> src.cols - 1 ? src.cols - 1 : px + 1);

		}
	}
	namedWindow("Original", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Original", src); // Show our image inside it.

	namedWindow("Resize", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Resize", dst); // Show our image inside it.
	waitKey(0); // Wait for a keystroke in the window

	return 0;
}
*/

/*
int main()
{
	Mat src = imread("C:\\Program Files\\opencv\\sources\\samples\\data\\lena.jpg", IMREAD_GRAYSCALE); // Read the file
	Mat dst(Size(src.rows, src.cols), CV_8UC1);
	Point2f srcTri[4];
	Point2f dstTri[4];

	srcTri[0] = Point2f(0, 0);
	srcTri[1] = Point2f(src.cols - 1, 0);
	srcTri[2] = Point2f(0, src.rows - 1);
	srcTri[3] = Point2f(src.cols - 1, src.rows - 1);
	dstTri[0] = Point2f(src.cols*0.0, src.rows*0.33);
	dstTri[1] = Point2f(src.cols*0.85, src.rows*0.25);
	dstTri[2] = Point2f(src.cols*0.15, src.rows*0.7);
	dstTri[3] = Point2f(src.cols*0.90, src.rows*0.8);
	//Mat warp_mat = getAffineTransform(srcTri, dstTri);
	//warpAffine(src, dst, warp_mat, dst.size());
	Mat warp_mat = getPerspectiveTransform(srcTri, dstTri);
	warpPerspective(src, dst, warp_mat, dst.size());

	namedWindow("Original", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Original", src); // Show our image inside it.

	namedWindow("Resize", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Resize", dst); // Show our image inside it.
	waitKey(0); // Wait for a keystroke in the window

	return 0;
}*/

/*
int main(int, char**)
{
	VideoCapture cap(0);
	if (!cap.isOpened())
		return -1;
	Mat src;
	Mat dst;
	Point2f srcTri[4];
	Point2f dstTri[4];


	namedWindow("src", 1);
	for (;;)
	{
		Mat frame;
		cap >> frame;
		cvtColor(frame, src, COLOR_BGR2GRAY);
		//GaussianBlur(src, src, Size(7, 7), 1.5, 1.5);
		//Canny(src, src, 0, 30, 3);
		srcTri[0] = Point2f(0, 0);
		srcTri[1] = Point2f(src.cols - 1, 0);
		srcTri[2] = Point2f(0, src.rows - 1);
		srcTri[3] = Point2f(src.cols - 1, src.rows - 1);
		dstTri[0] = Point2f(src.cols*0.0, src.rows*0.33);
		dstTri[1] = Point2f(src.cols*0.85, src.rows*0.25);
		dstTri[2] = Point2f(src.cols*0.15, src.rows*0.7);
		dstTri[3] = Point2f(src.cols*0.90, src.rows*0.7);
		Mat warp_mat = getPerspectiveTransform(srcTri, dstTri);
		warpPerspective(src, src, warp_mat, dst.size());
		imshow("src", src);
		waitKey(30);
	}
	return 0;
}*/

/*
int main(int, char**)
{
	Mat image1 = imread("D:\\Lecture\\4학년\\멀티미디어\\caltrain\\caltrain000.png", IMREAD_GRAYSCALE); // Read the file
	Mat image2 = imread("D:\\Lecture\\4학년\\멀티미디어\\caltrain\\caltrain001.png", IMREAD_GRAYSCALE); // Read the file

	int b_num = -1;
	if (image1.empty() || image2.empty()) // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	printf("%d %d\n", image1.rows, image1.cols);
	Mat dst = image1.clone();
	int size = 16;
	uchar block[16][16];
	copyMakeBorder(image1, image1, size, size, size, size, 0, 0);
	copyMakeBorder(image2, image2, size, size, size, size, 0, 0);
	printf("%d %d\n", image1.rows, image1.cols);
	for (int i = size; i < image1.rows-size; i+=size)
	{
		for (int j = size; j < image1.cols-size; j+=size)
		{
			b_num++;
			//store current block
			for (int k = 0; k < size; k++)
			{
				for (int t = 0; t < size; t++)
				{
					block[k][t] = image1.at<uchar>(i+k, j+t);
				}
			}
			int min = INT_MAX;
			int temp = 0;
			for (int k = i-15; k < i+15; k++)
			{
				for (int t = j-15; t <j+15; t++)
				{
					int sum = 0;
					for (int q = 0; q < size; q++)
					{
						for (int w = 0; w < size; w++)
						{
							sum += abs(block[q][w] - image2.at<uchar>(k+q, t+w));
						}
					}
					temp = sum / (size * size);
					if (temp < min)
						min = temp;
				}
			}
			for (int q = 0; q < size; q++)
			{
				for (int w = 0; w < size; w++)
				{
					image1.at<uchar>(i + q, j + w) = saturate_cast<uchar>(min*(1000/100));
				}
			}
		}
	}
	printf("%d\n", b_num);

	namedWindow("dst", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("dst", image1); // Show our image inside it.
	waitKey(0); // Wait for a keystroke in the window

	return 0;
}*/

/*
int main(int, char**)
{
	Mat left = imread("D:\\Lecture\\4학년\\멀티미디어\\tsukuba\\left.png", IMREAD_GRAYSCALE); // Read the file
	Mat right = imread("D:\\Lecture\\4학년\\멀티미디어\\tsukuba\\right.png", IMREAD_GRAYSCALE); // Read the file
	if (left.empty() || right.empty()) // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	printf("%d %d\n", left.rows, left.cols);
	int size = 16;
	int window = 5;
	int half = window / 2;
	int sum;
	int min;
	int disparity;
	Mat dst(Size(left.cols, left.rows), CV_8UC1);
	copyMakeBorder(left, left, size, size, size, size, 0, 0);
	copyMakeBorder(right, right, size, size, size, size, 0, 0);

	for (int i = size; i < left.rows -size; i++)
	{
		for (int j = size; j < left.cols - size; j++)
		{
			min = 10000;
			for (int k = 0; k < size; k++)
			{
				sum = 0;
				for (int m = -half; m < half; m++)
				{
					for (int n = -half; n < half; n++)
					{
						sum += pow(right.at<uchar>(i+m, j+n) - left.at<uchar>(i+m, j+n + k), 2);
					}
				}
				if (min > sum)
				{
					min = sum;
					disparity = k;
				}
			}

			dst.at<uchar>(i-size, j-size) = disparity * 16;
		}
	}
	namedWindow("dst", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("dst", dst); // Show our image inside it.
	waitKey(0); // Wait for a keystroke in the window

	return 0;
}*/

/*
int main(int, char**)
{
	Mat front = imread("D:\\Lecture\\4학년\\멀티미디어\\Girl_in_front_of_a_green_background.jpg", IMREAD_COLOR); // Read the file
	Mat back = imread("D:\\Lecture\\4학년\\멀티미디어\\bg.jpg", IMREAD_COLOR); // Read the file
	if (front.empty() || back.empty()) // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	resize(back, back, Size(front.cols, front.rows));
	Mat dst(Size(front.cols, front.rows), CV_8UC3);
	cvtColor(front, front, CV_BGR2YCrCb);
	cvtColor(back, back, CV_BGR2YCrCb);	
	int cr_key = front.at<Vec3b>(0, 0)[1], cb_key = front.at<Vec3b>(0, 0)[2];

	float distance = 0.0;
	float alpha = 0.0;
	float inner = 29.0, outer = 35.0;
	for (int i = 0; i < front.rows; i++)
	{
		for (int j = 0; j < front.cols; j++) {
			distance = sqrt(pow((cr_key - front.at<Vec3b>(i, j)[1]), 2) + pow((cb_key - front.at<Vec3b>(i, j)[2]), 2));
			if (distance < inner) //inner
			{
				alpha = 1.0f;
			}
			else if(distance > outer) // outer
			{
				alpha = 0.0f;
			}
			else {
				alpha = (distance - inner) / (outer - inner);
			}
			dst.at<Vec3b>(i, j)[0] = (1 - alpha)*front.at<Vec3b>(i, j)[0] + alpha*back.at<Vec3b>(i, j)[0];
			dst.at<Vec3b>(i, j)[1] = (1 - alpha)*front.at<Vec3b>(i, j)[1] + alpha*back.at<Vec3b>(i, j)[1];
			dst.at<Vec3b>(i, j)[2] = (1 - alpha)*front.at<Vec3b>(i, j)[2] + alpha*back.at<Vec3b>(i, j)[2];
		}
	}
	cvtColor(dst, dst, CV_YCrCb2BGR);
	namedWindow("dst", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("dst", dst); // Show our image inside it.
	waitKey(0); // Wait for a keystroke in the window

	return 0;
}*/

int main(int, char**)
{
	Mat src = imread("C:\\Program Files\\opencv\\sources\\samples\\data\\lena.jpg", IMREAD_GRAYSCALE); // Read the file
	if (src.empty()) // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	Mat dst = src.clone();
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 1; j < src.cols; j++)
		{
			dst.at<uchar>(i, j) = ((src.at<uchar>(i, j - 1) - src.at<uchar>(i, j)) + 255)/2;
		}
	}

	// Initialize parameters
	int histSize = 512; // bin size
	float range[] = { 0, 255 };
	const float *ranges[] = { range };

	Mat hist;
	calcHist(&dst, 1, 0, Mat(), hist, 1, &histSize, ranges, true, false);

	// Plot the histogram
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);
	Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(0, 0, 0));
	// Normalize the result to [ 0, histImage.rows]
	normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	for (int i = 1; i< histSize; i++)
	{
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
	}

	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 1; j < src.cols; j++)
		{
			dst.at<uchar>(i, j) = ((dst.at<uchar>(i, j - 1) + dst.at<uchar>(i, j)) - 255) / 2;
		}
	}


	namedWindow("dst", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("dst", histImage); // Show our image inside it.
	waitKey(0); // Wait for a keystroke in the window

	return 0;
}