#include "polygon_demo.hpp"
#include "opencv2/imgproc.hpp"
#include "iostream"
#include "stdlib.h"

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

void PolygonDemo::drawPolygon(Mat& frame, const std::vector<cv::Point>& vtx, bool closed)
{
    int i = 0;
    for (i = 0; i < (int)m_data_pts.size(); i++)
    {
        circle(frame, m_data_pts[i], 2, Scalar(255, 255, 255), CV_FILLED);
    }
    for (i = 0; i < (int)m_data_pts.size() - 1; i++)
    {
        line(frame, m_data_pts[i], m_data_pts[i + 1], Scalar(255, 255, 255), 1);
    }
    if (closed)
    {
        line(frame, m_data_pts[i], m_data_pts[0], Scalar(255, 255, 255), 1);
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
