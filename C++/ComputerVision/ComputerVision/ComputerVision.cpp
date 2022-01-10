#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>

#define DELAY 30
#define ESC_KEY 27
#define IMG_PATH "C:/Users/vanuc/Desktop/ComputerVision/C++/Samples/Mr-Robot.jpg"

using namespace std;
using namespace cv;

//const char* helper =
//"02_FaceDetection.exe <model_file> [<video>]\n\
//	\t<model_file> - model file name\n\
//	\t<video> - video file name (video stream will be taken from \n\
//				web-camera by default)\n\
//	";

int Sobel() {
	const char* initialWinName = "Initial Image",
		* xGradWinName = "Gradient in the direction Ox",
		* yGradWinName = "Gradient in the direction Oy",
		* gradWinName = "Gradient";
	int ddepth = CV_16S;
	double alpha = 0.5, beta = 0.5;
	Mat img, grayImg, xGrad, yGrad, xGradAbs, yGradAbs, grad;
	//if (argc < 2)
	//{
	//	printf("%s", helper);
	//	return 1;
	//}
	// загрузка изображения
	img = imread(IMG_PATH, 1);
	resize(img, img, Size(840, 525));
	// удаление шумов с помощью фильтра Гаусса
	GaussianBlur(img, img, Size(3, 3), 0, 0, BORDER_DEFAULT);
	// преобразование в оттенки серого
	cvtColor(img, grayImg, COLOR_BGR2GRAY);
	// вычисление производных по двум направлениям
	Sobel(grayImg, xGrad, ddepth, 1, 0); // по Ox
	Sobel(grayImg, yGrad, ddepth, 0, 1); // по Oy
	// преобразование градиентов в 8-битные беззнаковые
	convertScaleAbs(xGrad, xGradAbs);
	convertScaleAbs(yGrad, yGradAbs);
	// поэлементное вычисление взвешенной суммы двух массивов
	addWeighted(xGradAbs, alpha, yGradAbs, beta, 0, grad);

	// отображение результата
	namedWindow(initialWinName);
	namedWindow(xGradWinName);
	namedWindow(yGradWinName);
	namedWindow(gradWinName);
	imshow(initialWinName, img);
	imshow(xGradWinName, xGradAbs);
	imshow(yGradWinName, yGradAbs);
	imshow(gradWinName, grad);
	waitKey();
	// закрытие окон
	destroyAllWindows();
	// осовобождение памяти
	img.release();
	grayImg.release();
	xGrad.release();
	yGrad.release();
	xGradAbs.release();
	yGradAbs.release();
	return 0;

}

int MorphologyEx() {
	const char* initialWinName = "Initial Image",
		* morphologyOpenWinName = "MORPH_OPEN",
		* morphologyCloseWinName = "MORPH_CLOSE",
		* morphologyGradientWinName = "MORPH_GRADIENT",
		* morphologyTopHatWinName = "MORPH_TOPHAT",
		* morphologyBlackHatWinName = "MORPH_BLACKHAT";
	Mat img, morphologyOpenImg, morphologyCloseImg, morphologyGradientImg,
		morphologyTopHatImg, morphologyBlackHatImg, element;
	//if (argc < 2)
	//{
	//	printf("%s", helper);
	//	return 1;
	//}

	// загрузка изображения
	img = imread(IMG_PATH, 1);
	resize(img, img, Size(840, 525));

	// применение морфологических операций
	element = Mat();
	morphologyEx(img, morphologyOpenImg, MORPH_OPEN, element);
	morphologyEx(img, morphologyCloseImg, MORPH_CLOSE, element);
	morphologyEx(img, morphologyGradientImg, MORPH_GRADIENT, element);
	morphologyEx(img, morphologyTopHatImg, MORPH_TOPHAT, element);
	morphologyEx(img, morphologyBlackHatImg, MORPH_BLACKHAT, element);

	// отображение исходного изображения и результата выполнения операций
	namedWindow(initialWinName);
	namedWindow(morphologyOpenWinName);
	namedWindow(morphologyCloseWinName);
	namedWindow(morphologyGradientWinName);
	namedWindow(morphologyTopHatWinName);
	namedWindow(morphologyBlackHatWinName);
	imshow(initialWinName, img);
	imshow(morphologyOpenWinName, morphologyOpenImg);
	imshow(morphologyCloseWinName, morphologyCloseImg);
	imshow(morphologyGradientWinName, morphologyGradientImg);
	imshow(morphologyTopHatWinName, morphologyTopHatImg);
	imshow(morphologyBlackHatWinName, morphologyBlackHatImg);
	waitKey();

	// закрытие окон
	destroyAllWindows();
	// осовобождение памяти
	img.release();
	morphologyOpenImg.release();
	morphologyCloseImg.release();
	morphologyGradientImg.release();
	morphologyTopHatImg.release();
	morphologyBlackHatImg.release();
	return 0;

}

int Laplacian() {
	const char* initialWinName = "Initial Image",
		* laplacianWinName = "Laplacian";
	Mat img, grayImg, laplacianImg, laplacianImgAbs;
	int ddepth = CV_16S;
	//if (argc < 2)
	//{
	//	printf("%s", helper);
	//	return 1;
	//}
	// загрузка изображения
	img = imread(IMG_PATH, 1);
	resize(img, img, Size(840, 525));
	// удаление шумов с помощью фильтра Гаусса
	GaussianBlur(img, img, Size(3, 3), 0, 0, BORDER_DEFAULT);
	// преобразование в оттенки серого
	cvtColor(img, grayImg, COLOR_BGR2GRAY);
	// применение оператора Лапласа
	Laplacian(grayImg, laplacianImg, ddepth);
	convertScaleAbs(laplacianImg, laplacianImgAbs);

	// отображение результата
	namedWindow(initialWinName);
	namedWindow(laplacianWinName);
	imshow(initialWinName, img);
	imshow(laplacianWinName, laplacianImgAbs);
	waitKey();

	// закрытие окон
	destroyAllWindows();
	// осовобождение памяти
	img.release();
	grayImg.release();
	laplacianImg.release();
	laplacianImgAbs.release();
	return 0;

}

int Filter2D() {
	// константы для определения названия окон
	const char* initialWinName = "Initial Image", * resultWinName = "Filter2D";
	// константы для хранения ядра фильтра
	const float kernelData[] = { -0.1f, 0.2f, -0.1f,
								 0.2f, 3.0f,  0.2f,
								-0.1f, 0.2f, -0.1f };
	const Mat kernel(3, 3, CV_32FC1, (float*)kernelData);

	// объекты для хранения исходного и результирующего изображений
	Mat src, dst;

	// проверка аргументов командной строки
	//if (argc < 2)
	//{
	//	printf("%s", helper);
	//	return 1;
	//}

	// загрузка изображения
	src = imread(IMG_PATH, 1);
	resize(src, src, Size(840, 525));

	// применение фильтра
	filter2D(src, dst, -1, kernel);

	// отображение исходного изображения и результата применения фильтра
	namedWindow(initialWinName);
	imshow(initialWinName, src);
	namedWindow(resultWinName);
	imshow(resultWinName, dst);
	waitKey();

	// закрытие окон
	destroyAllWindows();
	// освобождение ресурсов
	src.release();
	dst.release();
	return 0;

}

int ErodeDilate() {
	const char* initialWinName = "Initial Image",
		* erodeWinName = "erode", * dilateWinName = "dilate";
	Mat img, erodeImg, dilateImg, element;
	//if (argc < 2)
	//{
	//	printf("%s", helper);
	//	return 1;
	//}
	// загрузка изображения
	img = imread(IMG_PATH, 1);
	resize(img, img, Size(840, 525));
	// вычисление эрозии и дилатации
	element = Mat();
	erode(img, erodeImg, element);
	dilate(img, dilateImg, element);
	// отображение исходного изображения и результата
	// применения морфологических операций "эрозия" и "дилатация"
	namedWindow(initialWinName);
	namedWindow(erodeWinName);
	namedWindow(dilateWinName);
	imshow(initialWinName, img);
	imshow(erodeWinName, erodeImg);
	imshow(dilateWinName, dilateImg);
	waitKey();

	// закрытие окон
	destroyAllWindows();
	// освобождение ресурсов
	img.release();
	erodeImg.release();
	dilateImg.release();
	return 0;

}

int EqualizeHist() {
	const char* initialWinName = "Initial Image",
		* equalizedWinName = "Equalized Image";
	Mat img, grayImg, equalizedImg;
	//if (argc < 2)
	//{
	//	printf("%s", helper);
	//	return 1;
	//}
	// загрузка изображения
	img = imread(IMG_PATH, 1);
	resize(img, img, Size(840, 525));
	// преобразование в оттенки серого
	cvtColor(img, grayImg, COLOR_BGR2GRAY);
	// выравнивание гистограммы
	equalizeHist(grayImg, equalizedImg);

	// отображение исходного изображения и гистограмм
	namedWindow(initialWinName);
	namedWindow(equalizedWinName);
	imshow(initialWinName, grayImg);
	imshow(equalizedWinName, equalizedImg);
	waitKey();

	// закрытие окон
	destroyAllWindows();
	// осовобождение памяти
	img.release();
	grayImg.release();
	equalizedImg.release();
	return 0;

}

int BlurHist() {
	const char* initialWinName = "Initial Image",
		* histWinName = "Histogram";
	Mat img, bgrChannels[3], bHist, gHist, rHist, histImg;
	int kBins = 256; // количество бинов гистограммы
	float range[] = { 0.0f, 256.0f }; // интервал изменения значений бинов
	const float* histRange = { range };
	bool uniform = true; // равномерное распределение интервала по бинам
	bool accumulate = false; // запрет очищения перед вычислением гистограммы
	int histWidth = 512, histHeight = 400; // размеры для отображения гистограммы
	int binWidth = cvRound((double)histWidth / kBins); // количество пикселей на бин
	int i, kChannels = 3;
	Scalar colors[] = { Scalar(255, 0, 0), Scalar(0, 255, 0), Scalar(0, 0, 255) };
	//if (argc < 2)
	//{
	//	printf("%s", helper);
	//	return 1;
	//}
	// загрузка изображения
	img = imread(IMG_PATH, 1);
	// выделение каналов изображения
	split(img, bgrChannels);
	calcHist(&bgrChannels[0], 1, 0, Mat(), bHist, 1, &kBins,
		&histRange, uniform, accumulate);
	calcHist(&bgrChannels[1], 1, 0, Mat(), gHist, 1, &kBins,
		&histRange, uniform, accumulate);
	calcHist(&bgrChannels[2], 1, 0, Mat(), rHist, 1, &kBins,
		&histRange, uniform, accumulate);


	histImg = Mat(histHeight, histWidth, CV_8UC3, Scalar(0, 0, 0));

	normalize(bHist, bHist, 0, histImg.rows,
		NORM_MINMAX, -1, Mat());
	normalize(gHist, gHist, 0, histImg.rows,
		NORM_MINMAX, -1, Mat());
	normalize(rHist, rHist, 0, histImg.rows,
		NORM_MINMAX, -1, Mat());

	for (i = 1; i < kBins; i++)
	{
		line(histImg, Point(binWidth * (i - 1), histHeight - cvRound(bHist.at<float>(i - 1))),
			Point(binWidth * i, histHeight - cvRound(bHist.at<float>(i))),
			colors[0], 2, 8, 0);
		line(histImg, Point(binWidth * (i - 1), histHeight - cvRound(gHist.at<float>(i - 1))),
			Point(binWidth * i, histHeight - cvRound(gHist.at<float>(i))),
			colors[1], 2, 8, 0);
		line(histImg, Point(binWidth * (i - 1), histHeight - cvRound(rHist.at<float>(i - 1))),
			Point(binWidth * i, histHeight - cvRound(rHist.at<float>(i))),
			colors[2], 2, 8, 0);
	}
	namedWindow(initialWinName);
	namedWindow(histWinName);
	imshow(initialWinName, img);
	imshow(histWinName, histImg);
	waitKey();

	// закрытие окон
	destroyAllWindows();
	// осовобождение памяти
	img.release();
	for (i = 0; i < kChannels; i++)
	{
		bgrChannels[i].release();
	}
	bHist.release();
	gHist.release();
	rHist.release();
	histImg.release();
	return 0;

}

int Blur() {
	const char* initialWinName = "Initial Image",
		* blurWinName = "blur";
	Mat img, blurImg;
	//if (argc < 2)
	//{
	//	printf("%s", helper);
	//	return 1;
	//}

	// загрузка изображения
	img = imread(IMG_PATH, 1);
	resize(img, img, Size(840, 525));
	// применение операции размытия
	blur(img, blurImg, Size(5, 5));

	// отображение исходного изображения и результата размытия
	namedWindow(initialWinName);
	namedWindow(blurWinName);
	imshow(initialWinName, img);
	imshow(blurWinName, blurImg);
	waitKey();

	// закрытие окон
	destroyAllWindows();
	// освобождение ресурсов
	img.release();
	blurImg.release();
	return 0;

}

int CannyEdges() {
	const char* srcWinName = "original", * contourWinName = "edges";
	namedWindow(srcWinName, 1);
	namedWindow(contourWinName, 1);
	//load original image
	Mat src = imread(IMG_PATH, 1);
	if (src.data == 0)
	{
		printf("Incorrect image name or format.\n");
		return 1;
	}
	resize(src, src, Size(840, 525));
	//create a single channel 1 byte image (i.e. gray-level image)
	//make a copy of the original image to draw the detected contour
	Mat copy = src.clone() * 0;
	Mat gray, grayThresh;
	cvtColor(src, gray, COLOR_BGR2GRAY);
	threshold(gray, grayThresh, 120, 255, THRESH_BINARY);
	//find the contour
	vector<vector<Point>> contours;
	findContours(grayThresh, contours, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
	// iterate through all the top-level contours,
	// draw each connected component with its own random color
	Scalar color(128, 128, 128);
	drawContours(copy, contours, -1, color, 1);
	imshow(contourWinName, copy);
	imshow(srcWinName, src);
	waitKey(0);
	gray.release();
	grayThresh.release();
	copy.release();
	src.release();
	return 0;
}

int SearchFaces() {
	const char* winName = "edges+face";
	//char* modelFileName = 0, * videoFileName = 0;
	const char* modelFileName = "C:/Users/vanuc/Desktop/ComputerVision/C++/Data/lbpcascade_frontalface_improved.xml", * videoFileName = 0;
	int i, width, height;
	char key = -1;
	Mat image, gray;
	VideoCapture capture;
	std::vector<Rect> objects;

	//if (argcN < 2)
	//{
	//	printf("%s", helper);
	//	return 1;
	//}
	//if (argcN > 2)
	//{
	//	videoFileName = argv[2];
	//}
	// создание классификатора и загрузка модели
	CascadeClassifier cascade;
	cascade.load(modelFileName);
	// загрузка видеофайла или перехват видеопотока
	if (videoFileName == 0)
	{
		capture.open(0);
	}
	else
	{
		capture.open(videoFileName);
	}
	if (!capture.isOpened())
	{
		printf("Incorrect capture name.\n");
		return 1;
	}

	// создание окна для отображения видео
	namedWindow(winName);
	// получение кадра видеопотока
	capture >> image;
	while (image.data != 0 && key != ESC_KEY)
	{
		cvtColor(image, gray, COLOR_BGR2GRAY);
		cascade.detectMultiScale(gray, objects);
		for (i = 0; i < objects.size(); i++)
		{
			rectangle(image, Point(objects[i].x, objects[i].y),
				Point(objects[i].x + objects[i].width, objects[i].y + objects[i].height),
				CV_RGB(255, 0, 0), 2);
		}
		imshow(winName, image);
		key = waitKey(DELAY);
		capture >> image;
		gray.release();
		objects.clear();
	}
	capture.release();
	return 0;
}

int main()
{
	//Sobel();
	//MorphologyEx();
	//Laplacian();
	//Filter2D();
	//ErodeDilate();
	//EqualizeHist();
	//BlurHist();
	//Blur();
	//SearchFaces();
	//CannyEdges();
}

