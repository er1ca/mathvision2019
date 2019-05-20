#include "polygon_demo.hpp"
#include "opencv2/imgproc.hpp"
#include "iostream"
#include "stdlib.h"
#include <list>
#include <algorithm>

using namespace std;
using namespace cv;

PolygonDemo::PolygonDemo()
{
    m_data_ready = false;
}

PolygonDemo::~PolygonDemo()
{
}

void PolygonDemo::refreshWindow()
{
    Mat frame = Mat::zeros(480, 640, CV_8UC3);
    if (!m_data_ready)
        putText(frame, "Input data points (double click: finish)", Point(10, 470), FONT_HERSHEY_SIMPLEX, .5, Scalar(0, 148, 0), 1);

    drawPolygon(frame, m_data_pts, m_data_ready);
    if (m_data_ready)
    {
        // polygon area
        if (m_param.compute_area)
        {
            int area = polyArea(m_data_pts);
            char str[100];
            sprintf_s(str, 100, "Area = %d", area);
            putText(frame, str, Point(25, 25), FONT_HERSHEY_SIMPLEX, .8, Scalar(0, 255, 255), 1);
        }

        // pt in polygon
        if (m_param.check_ptInPoly)
        {
            for (int i = 0; i < (int)m_test_pts.size(); i++)
            {
                if (ptInPolygon(m_data_pts, m_test_pts[i]))
                {
                    circle(frame, m_test_pts[i], 2, Scalar(0, 255, 0), CV_FILLED);
                }
                else
                {
				    circle(frame, m_test_pts[i], 2, Scalar(128, 128, 128), CV_FILLED);
                }
            }
        }

        // homography check
        if (m_param.check_homography && m_data_pts.size() == 4)
        {
            // rect points
            int rect_sz = 100;
            vector<Point> rc_pts;
            rc_pts.push_back(Point(0, 0));
            rc_pts.push_back(Point(0, rect_sz));
            rc_pts.push_back(Point(rect_sz, rect_sz));
            rc_pts.push_back(Point(rect_sz, 0));
            rectangle(frame, Rect(0, 0, rect_sz, rect_sz), Scalar(255, 255, 255), 1);

            // draw mapping
            char* abcd[4] = { "A", "B", "C", "D" };
            for (int i = 0; i < 4; i++)
            {
                line(frame, rc_pts[i], m_data_pts[i], Scalar(255, 0, 0), 1);
                circle(frame, rc_pts[i], 2, Scalar(0, 255, 0), CV_FILLED);
                circle(frame, m_data_pts[i], 2, Scalar(0, 255, 0), CV_FILLED);
                putText(frame, abcd[i], m_data_pts[i], FONT_HERSHEY_SIMPLEX, .8, Scalar(0, 255, 255), 1);
            }

            // check homography
            int homo_type = classifyHomography(rc_pts, m_data_pts);
            char type_str[100];
            switch (homo_type)
            {
            case NORMAL:
                sprintf_s(type_str, 100, "normal");
                break;
            case CONCAVE:
                sprintf_s(type_str, 100, "concave");
                break;
            case TWIST:
                sprintf_s(type_str, 100, "twist");
                break;
            case REFLECTION:
                sprintf_s(type_str, 100, "reflection");
                break;
            case CONCAVE_REFLECTION:
                sprintf_s(type_str, 100, "concave reflection");
               break;
            }

            putText(frame, type_str, Point(15, 125), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 255), 1);
        }

        // fit circle
        if (m_param.fit_circle)
        {
            Point2d center;
            double radius = 0;
            bool ok = fitCircle(m_data_pts, center, radius);
            if (ok)
            {
                circle(frame, center, (int)(radius + 0.5), Scalar(0, 255, 0), 1);
                circle(frame, center, 2, Scalar(0, 255, 0), CV_FILLED);
            }
        }
		// fit Ellipse
		if (m_param.fit_ellipse)
		{
			Point2d  m;
			Point2d  v;
			double theta;
			bool ok = fitEllipse(m_data_pts, m, v, theta);

			if (ok){
				// size vx,vy? 
				ellipse(frame, m, Size((int)v.x,(int)v.y), theta, 0, 360, Scalar(0, 255, 0), 1, 8);
			}
		}
		// HW#9 draw Line
		if (m_param.draw_line)
		{
			Point2d point1;
			Point2d point2;
			
			String fx1 = "y = ax + b";
			putText(frame, fx1, Point(15, 35), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 1);
			drawLine(m_data_pts, point1, point2);
			line(frame, point1, point2, Scalar(0, 255, 0),3);
			
			String fx2 = "ax +by + c = 0";
			putText(frame, fx2, Point(15, 55), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 0), 1);
			drawLine_SVD(m_data_pts, point1, point2);
			line(frame, point1, point2, Scalar(255, 255, 0),2);
		}
		// HW#10 fit Line with Cauchy Weighted
		if (m_param.fit_line)
		{
			Point2d point1;
			Point2d point2;
			Mat residual = Mat::zeros(m_data_pts.size(), 1, CV_32FC1);
			String fx1 = "Cauchy Weighted LS y = ax + b";
			putText(frame, fx1, Point(15, 75), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 255), 1);
			bool flag = true;
			//int iteration = 0 
			for (int i = 0; i < 10; i++){
				drawLine_CauchyWeigtedLS(m_data_pts, point1, point2, flag, residual, i);
				line(frame, point1, point2, Scalar(0, 50, 50+20*i));
				cout << i << endl;
			}
			drawLine_CauchyWeigtedLS(m_data_pts, point1, point2, flag, residual, 10);
			line(frame, point1, point2, Scalar(50, 0, 255),2);
		}
		// HW#11 fit line with RANSAC
		if (m_param.fit_RANSAC)
		{
			Point2d point1;
			Point2d point2;
			Mat model = Mat::zeros(3, 1, CV_32FC1); // saved model

			/*
			String fx1 = "y = ax + b";
			putText(frame, fx1, Point(15, 35), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 0, 255), 1);
			drawLine_RANSAC(m_data_pts, point1, point2);
			line(frame, point1, point2, Scalar(255, 0, 255), 1);
			*/
			
			String fx2 = "ax +by + c = 0";
			putText(frame, fx2, Point(15, 25), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(50, 0, 255), 1);
			drawLine_RANSAC_1(m_data_pts, point1, point2);
			line(frame, point1, point2, Scalar(50, 0, 255), 1);
		}

    }

    imshow("PolygonDemo", frame);
}

// return the area of polygon
int PolygonDemo::polyArea(const std::vector<cv::Point>& vtx)
{
	int S = 0;
	int num = vtx.size();
	Point v[2];
	for (int i = 0; i < num-2; i++){
		//next point index
		int i2 = i + 1; 
		int i3 = i + 2; 

		//define vectors for every points
		v[0] = vtx[i] - vtx[i2];
		v[1] = vtx[i2] - vtx[i3];
		S += v[0].cross(v[1]) / 2;
	}
	/*
	for (int i = 0; i < num; i++){
		printf("vtx[%d]=(%d, %d)\r\n", i, vtx[i].x, vtx[i].y);
	}
	*/
    return abs(S);
}

// return true if pt is interior point
bool PolygonDemo::ptInPolygon(const std::vector<cv::Point>& vtx, Point pt)
{
	bool flag = false;
	for (int i = 0; i < vtx.size() - 1; i++){
		if (pt.cross(vtx[i]) < 0)
			flag = true;
		else
			flag = false;
	}
    return flag;
}

// return homography type: NORMAL, CONCAVE, TWIST, REFLECTION, CONCAVE_REFLECTION
int PolygonDemo::classifyHomography(const std::vector<cv::Point>& pts1, const std::vector<cv::Point>& pts2)
{
	int Flag = 0;
	int nMinus = 0;
	int check = 1;
	
	if (pts1.size() != 4 || pts2.size() != 4) return -1;
	Point v[2]; //vector
	int c[4];
	for (int i = 0; i < 4; i++){
		//next point index
		int start = i - 1; if (start < 0) start = 3;
		int end = i + 1; if (end > 3) end = 0;

		//define vectors for every points
		v[0] = pts2[i] - pts2[start];
		v[1] = pts2[i] - pts2[end];
		//printf("%d , %d\r\n", v[0],v[1]);
		
		//calculate cross prodoct for every vectors
		c[i] = v[0].cross(v[1]);
		//printf("c[%d] : %d\r\n", i, c[i]);

		if (c[i] < 0) nMinus++;
		
		check = check * (c[i]/100);
	}
	printf("nMinus : %d, check : %d\r\n", nMinus, check);

	//check rectangle types
	if (nMinus == 3) Flag = CONCAVE_REFLECTION;
	if (nMinus == 1) Flag = CONCAVE;
	if (nMinus == 4) Flag = REFLECTION;
	if (nMinus == 0) Flag = NORMAL;
	if (nMinus == 2) Flag = TWIST; 
		
	printf("Flag : %d", Flag);
		
	return Flag;
}


// estimate a circle that best approximates the input points and return center and radius of the estimate circle
bool PolygonDemo::fitCircle(const std::vector<cv::Point>& pts, cv::Point2d& center, double& radius)
{
	int n = (int)pts.size();
	if (n < 3) return false;
	Mat J = Mat::ones(n, 3, CV_64F);
	Mat Y = Mat::ones(n, 1, CV_64F);
	Mat X = Mat::ones(3, 1, CV_64F);

	for (int i = 0; i < n; i ++){
		double x = pts[i].x;
		double y = pts[i].y;
		
		J.at<double>(i, 0) = -2 * x;
		J.at<double>(i, 1) = -2 * y;
		//J.at<double>(i, 2) = 1;
		Y.at<double>(i, 0) = -pow(x, 2) - pow(y, 2);
	}
	//cout << J << endl;

	
	/*
	Mat W, U, Vt;
	SVD::compute(J, W, U, Vt, SVD::FULL_UV);
	cout << W << endl; // Singular value
	*/
	
	//cout << J.t() << endl;
	Mat pseudoInvJ;
	invert(J, pseudoInvJ, DECOMP_SVD);;
	//cout << pseudoInvJ << endl;
	X = pseudoInvJ*Y;
	//cout << X << endl;

	double a = X.at<double>(0, 0);
	double b = X.at<double>(1, 0);
	double c = X.at<double>(2, 0);

	center.x = a;
	center.y = b;
	radius = sqrt(pow(a,2) + pow(b,2) - c);
	
	printf("center (%d, %d) \n", center.x, center.y);
	printf("radius : %d \n", radius);

	cout << "JX : \n" << J*X << endl;
	cout << "Y : \n" << Y << endl;

	return true;
}

// estimate a ellipse that best approximates the input points and return center and radius of the estimate circle
bool PolygonDemo::fitEllipse(const std::vector<cv::Point>& pts, cv::Point2d& m, cv::Point2d& v, double& theta)
{
	int n = (int)pts.size();
	if (n < 3) return false;

	Mat J = Mat::ones(n, 6, CV_64F);
	Mat X = Mat::ones(6, 1, CV_64F);
	
	for (int i = 0; i < pts.size(); i++) 
	{
		double x = pts[i].x;
		double y = pts[i].y;

		J.at<double>(i, 0) = pow(x,2);
		J.at<double>(i, 1) = x*y;
		J.at<double>(i, 2) = pow(y,2);
		J.at<double>(i, 3) = x;
		J.at<double>(i, 4) = y;
		J.at<double>(i, 5) = 1;
	}

	Mat W, U, Vt, svd;
	SVD::compute(J, W, U, Vt, SVD::FULL_UV);
	cout << W << endl; // Singular value

	transpose(Vt, svd);
	
	for (int i = 0; i < 6; i++)	
		X.at<double>(i, 0) = svd.at<double>(i, 5);
	//cout << X << endl;
	

	double a = X.at<double>(0, 0);
	double b = X.at<double>(1, 0);
	double c = X.at<double>(2, 0);
	double d = X.at<double>(3, 0);
	double e = X.at<double>(4, 0);
	double f = X.at<double>(5, 0);

	theta = atan2(b, a - c) / 2 ; 

	//center x, y
	m.x = (2 * c * d - b * e) / (pow(b, 2) - 4 * a * c);
	m.y = (2 * a * e - b * d) / (pow(b, 2) - 4 * a * c);
	cout << "m : " << m << endl;

	//w = v.x, h = v.y
	v.x = sqrt((a * pow(m.x, 2) + b * m.x * m.y + c * pow(m.y, 2) - f) 
		/ (a * pow(cos(theta), 2) + b * cos(theta) * sin(theta) + c * pow(sin(theta), 2)));
	v.y = sqrt((a * pow(m.x, 2) + b * m.x * m.y + c * pow(m.y, 2) - f) 
		/ (a * pow(sin(theta), 2) - b * cos(theta) * sin(theta) + c * pow(cos(theta), 2)));
	cout << "v : " << v << endl;
	theta *= 180 / 3.14;

	cout << "JX : \n" << J*X << endl;

	return true;
}
// HW#9 draw Line
bool PolygonDemo::drawLine(const std::vector<cv::Point>& pts, cv::Point2d& point1, cv::Point2d& point2)
{
	int n = (int)pts.size();
	if (n < 2) return false;

	Mat J = Mat::zeros(n, 2, CV_32FC1);
	Mat X = Mat::zeros(2, 1, CV_32FC1);
	Mat Y = Mat::zeros(n, 1, CV_32FC1);
	Mat Jpinv;

	for (int i = 0; i < n; i++){
		J.at<float>(i, 0) = pts[i].x;
		J.at<float>(i, 1) = 1;
		Y.at<float>(i, 0) = pts[i].y;
	}

	Mat d_svd, u, svd_t, svd;
	SVD::compute(J, d_svd, u, svd_t, SVD::FULL_UV);
	invert(J, Jpinv, DECOMP_SVD);
	X = Jpinv*Y;

	point1.x = 0; 
	point2.x = 640;
	point1.y = X.at<float>(0, 0)*point1.x + X.at<float>(1, 0);
	point2.y = X.at<float>(0, 0)*point2.x + X.at<float>(1, 0);
	return true;

}
// HW#9 draw Line
bool PolygonDemo::drawLine_SVD(const std::vector<cv::Point>& pts, cv::Point2d& point1, cv::Point2d& point2)
{
	int n = (int)pts.size();
	//if (n < 2) return false;

	Mat A = Mat::zeros(n, 3, CV_32FC1); 
	Mat X = Mat::zeros(3, 1, CV_32FC1);
	Mat pinvA;

	for (int i = 0; i < n; i++){
		A.at<float>(i, 0) = pts[i].x;
		A.at<float>(i, 1) = pts[i].y;
		A.at<float>(i, 2) = 1;  
	}

	Mat d_svd, u, svd_t, svd;
	SVD::compute(A, d_svd, u, svd_t, SVD::FULL_UV);
	transpose(svd_t, svd);
	for (int i = 0; i < 3; i++){
		X.at<float>(i, 0) = svd.at<float>(i, 2);
	}

	point1.x = 0;
	point2.x = 640;

	point1.y = -(X.at<float>(0, 0) / X.at<float>(1, 0))*point1.x - (X.at<float>(2, 0) / X.at<float>(1, 0));
	point2.y = -(X.at<float>(0, 0) / X.at<float>(1, 0))*point2.x - (X.at<float>(2, 0) / X.at<float>(1, 0));
	
	return true;
}

// HW#10 fit Line
bool PolygonDemo::drawLine_CauchyWeigtedLS(const std::vector<cv::Point>& pts, cv::Point2d& point1, cv::Point2d& point2, bool flag, cv::Mat& residual, int iteration)
{
	int n = (int)pts.size();
	Mat A = Mat::zeros(n, 2, CV_32FC1); // points
	Mat X = Mat::zeros(n, 1, CV_32FC1); // 
	//Mat Y = Mat::zeros(n, 1, CV_32FC1);
	Mat p = Mat::zeros(2, 1, CV_32FC1); // param
	Mat W = Mat::zeros(n, n, CV_32FC1); // cauchy weights
	Mat pinvA;

	
	//initialization at iteration 0
	if (iteration == 0) {
		for (int i = 0; i < n; i++){
			A.at<float>(i, 0) = pts[i].x;
			A.at<float>(i, 1) = 1;
			X.at<float>(i, 0) = pts[i].y;
		}
		invert(A, pinvA, DECOMP_SVD);
		p = pinvA * X;
		cout << residual << endl;
	}
	else{
		for (int i = 0; i < n; i++){
			A.at<float>(i, 0) = pts[i].x;
			A.at<float>(i, 1) = 1;
			X.at<float>(i, 0) = pts[i].y;
			W.at<float>(i, i) = 1 / (residual.at<float>(i, 0) / 1.3998 + 1); //Cauchy weight function 
		}
		//||W^1/2(Ap-x)||^2을 최소화시키는 해 
		p = (A.t() * W * A).inv(1) * A.t() * W * X;
	}
	
	residual = A*p - X;
	point1.x = 0;
	point2.x = 640;

	point1.y = p.at<float>(0, 0) * point1.x + p.at<float>(1, 0);
	point2.y = p.at<float>(0, 0) * point2.x + p.at<float>(1, 0);

	return true; 


}

//HW#11 fit line with RANSAC
bool PolygonDemo::drawLine_RANSAC(const std::vector<cv::Point> & pts, cv::Point2d& point1, cv::Point2d& point2)
{
	int n = (int)pts.size();
	if (n < 2) return false;

	const int s = 2;
	float th = 100.0;
	int iteration = 6;

	//cout << iteration << endl;
	
	Mat model = Mat::zeros(5, 3, CV_32FC1); // saved model cnt, a, b 
	Mat X = Mat::zeros(n, 2, CV_32FC1); // x values
	Mat Y = Mat::zeros(n, 1, CV_32FC1); // y values

	//model y = ax + b
	for (int i = 0; i < n; i++){
		X.at<float>(i, 0) = pts[i].x;
		X.at<float>(i, 1) = 1;
		Y.at<float>(i, 0) = pts[i].y;
	}
	//cout << X << endl;
	//cout << Y << endl;

	for (int j = 0; j < iteration; j++){
		Mat P = Mat::zeros(2, 1, CV_32FC1); // param a,b
		Mat pinvS;

		Mat S_x = Mat::zeros(s, 2, CV_32FC1); // Sample points
		Mat S_y = Mat::zeros(s, 1, CV_32FC1); // Sample points
		Mat residual = Mat::zeros(n, 1, CV_32FC1);

		int ran[s];
		for (int i = 0; i < s; i++)	ran[i] = (rand() % n);

		//Random Sample
		for (int i = 0; i < s; i++){
			S_x.at<float>(i, 0) = X.at<float>(ran[i], 0);
			S_x.at<float>(i, 1) = 1;
			S_y.at<float>(i, 1) = Y.at<float>(ran[i], 0);
			cout << S_x << endl;
			cout << S_y << endl;
		}

		invert(S_x, pinvS, DECOMP_SVD);
		P = pinvS * S_y;
		//cout << P << endl;

		residual = abs(Y - X*P);
		cout << residual << endl;

		int cnt = 0;
		for (int i = 0; i < n; i++){
			if (residual.at<float>(i, 0) < th){
				cnt++;
			}
		}
		model.at<float>(j, 0) = cnt;
		model.at<float>(j, 1) = P.at<float>(0, 0);
		model.at<float>(j, 2) = P.at<float>(0, 1);

		
	}
	int cur_max = 0;
	int max_index = 0;
	for (int i = 0; i < iteration; i++){
		if (model.at<float>(i, 0)>cur_max){
			max_index = i;
			cur_max = model.at<float>(i, 0);
		}
	}
	

	cout << model.at<float>(max_index, 1) << model.at<float>(max_index, 2) << endl;

	point1.x = 0;
	point2.x = 640;
	
	point1.y = model.at<float>(max_index, 1)*point1.x + model.at<float>(max_index, 2);
	point2.y = model.at<float>(max_index, 1)*point2.x + model.at<float>(max_index, 2);

	cout << point1 << endl;
	cout << point2 << endl;

	return true;

}

//ax + by + c = 0 모델 
bool PolygonDemo::drawLine_RANSAC_1(const std::vector<cv::Point> & pts, cv::Point2d& point1, cv::Point2d& point2){

	int n = (int)pts.size();
	if (n < 2) return false;

	const int s = 2;
	float th = 20.0;
	int iteration = 6;

	float a, b, c = 0;
	Mat A = Mat::zeros(n, 3, CV_32FC1); // points
	
	//Mat A
	for (int i = 0; i < n; i++){
		A.at<float>(i, 0) = pts[i].x;
		A.at<float>(i, 1) = pts[i].y;
		A.at<float>(i, 2) = 1;
	}
	//cout << A << endl;
	
	Mat model = Mat::zeros(iteration, 4, CV_32FC1); // saved model cnt, a, b, c
	Mat best_model = Mat::zeros(1, 4, CV_32FC1); // best model cnt, a, b, c

	for (int i = 0; i < iteration; i++){
		cout << i << endl;
		Mat P = Mat::zeros(3, 1, CV_32FC1); // param a,b,c
		Mat S = Mat::zeros(s, 3, CV_32FC1); // Sample points
		Mat sampleV = Mat::zeros(1, 2, CV_32FC1);
		Mat residual = Mat::zeros(n, 1, CV_32FC1);
		Mat pinvA;

		//Random Sample
		int ran[s];
		for (int i = 0; i < s; i++){
			int tmp = (rand() % n);
			cout << tmp << endl;
			for (int k = 0; k < i; k++) if (ran[k] == tmp) tmp = (rand()%n);
			ran[i] = tmp;
		}
		cout << ran << endl;
		for (int i = 0; i < s; i++){
			S.at<float>(i, 0) = A.at<float>(ran[i], 0);
			S.at<float>(i, 1) = A.at<float>(ran[i], 1);
			S.at<float>(i, 2) = 1;
			//cout << S << endl;
		}

		//get param a,b,c
		Mat d_svd, u, svd_t, svd;
		SVD::compute(S, d_svd, u, svd_t, SVD::FULL_UV);
		transpose(svd_t, svd);

		//set param for Mat X
		for (int i = 0; i < 3; i++){
			P.at<float>(i, 0) = svd.at<float>(i, 2);
		}
		//cout << P << endl; //a, b, c

		a = P.at<float>(0, 0);
		b = P.at<float>(0, 1);
		c = P.at<float>(0, 2);
		//get residual
		int cnt = 0;
		for (int i = 0; i < n; i++){
			residual.at<float>(i, 0) = abs((a * A.at<float>(i, 0) + b * A.at<float>(i, 1) + c) / sqrt(a*a + b*b));
			
			if (residual.at<float>(i, 0) < th){
				model.at<float>(i, 0) = cnt++;
				model.at<float>(i, 1) = a;
				model.at<float>(i, 2) = b;
				model.at<float>(i, 3) = c;
			}
		}
		cout << residual << endl;
	}

	int cur_max = 0;
	int max_index = 0;
	for (int i = 0; i < iteration; i++){
		if (model.at<float>(i, 0)>cur_max){
			max_index = i;
			cur_max = model.at<float>(i, 0);
		}
	}

	a = model.at<float>(max_index, 1);
	b = model.at<float>(max_index, 2);
	c = model.at<float>(max_index, 3);

	//cout << cnt << endl;
	//cout << residual << endl;

	point1.x = 0;
	point2.x = 640;

	point1.y = -(a / b)*point1.x - (c / b);
	point2.y = -(a / b)*point2.x - (c / b);

	return true;
}

void PolygonDemo::drawPolygon(Mat& frame, const std::vector<cv::Point>& vtx, bool closed)
{
    int i = 0;
    for (i = 0; i < (int)m_data_pts.size(); i++)
    {
        circle(frame, m_data_pts[i], 2, Scalar(255, 255, 255), CV_FILLED);
    }
    for (i = 0; i < (int)m_data_pts.size() - 1; i++)
    {
        //line(frame, m_data_pts[i], m_data_pts[i + 1], Scalar(255, 255, 255), 1);
    }
    if (closed)
    {
        //line(frame, m_data_pts[i], m_data_pts[0], Scalar(255, 255, 255), 1);
    }
}

void PolygonDemo::handleMouseEvent(int evt, int x, int y, int flags)
{
    if (evt == CV_EVENT_LBUTTONDOWN)
    {
        if (!m_data_ready)
        {
            m_data_pts.push_back(Point(x, y));
        }
        else
        {
            m_test_pts.push_back(Point(x, y));
        }
        refreshWindow();
    }
    else if (evt == CV_EVENT_LBUTTONUP)
    {
    }
    else if (evt == CV_EVENT_LBUTTONDBLCLK)
    {
        m_data_ready = true;
        refreshWindow();
    }
    else if (evt == CV_EVENT_RBUTTONDBLCLK)
    {
    }
    else if (evt == CV_EVENT_MOUSEMOVE)
    {
    }
    else if (evt == CV_EVENT_RBUTTONDOWN)
    {
        m_data_pts.clear();
        m_test_pts.clear();
        m_data_ready = false;
        refreshWindow();
    }
    else if (evt == CV_EVENT_RBUTTONUP)
    {
    }
    else if (evt == CV_EVENT_MBUTTONDOWN)
    {
    }
    else if (evt == CV_EVENT_MBUTTONUP)
    {
    }

    if (flags&CV_EVENT_FLAG_LBUTTON)
    {
    }
    if (flags&CV_EVENT_FLAG_RBUTTON)
    {
    }
    if (flags&CV_EVENT_FLAG_MBUTTON)
    {
    }
    if (flags&CV_EVENT_FLAG_CTRLKEY)
    {
    }
    if (flags&CV_EVENT_FLAG_SHIFTKEY)
    {
    }
    if (flags&CV_EVENT_FLAG_ALTKEY)
    {
    }
}
