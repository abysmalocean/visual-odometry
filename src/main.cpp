/*
CMSC 591 Slam Project
First Part, VO 
Author: Liang Xu
Data: 04/2019
Contact: liangxuav@gmail.com
*/

#include <iostream>
#include <../include/utility.h>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>

using std::cout; 
using std::endl;

void poseEstimation3D3D
(const std::vector<cv::Point3d>& pts1, 
 const std::vector<cv::Point3d>& pts2,
 std::vector<double> &R, std::vector<double>& t)
{
    cv::Point3d p1, p2; 
    int N = pts1.size(); 
    for (int i = 0; i < N; ++i)
    {
        p1 += pts1[i]; 
        p2 += pts2[i]; 
    }
    p1 = cv::Point3d(cv::Vec3d(p1) / N); 
    p2 = cv::Point3d(cv::Vec3d(p2) / N); 
    std::vector<cv::Point3i>     q1 ( N ), q2 ( N ); // remove the center
    for ( int i=0; i<N; i++ )
    {
        q1[i] = pts1[i] - p1;
        q2[i] = pts2[i] - p2;
    }
    // compute q1 * q2 ^ t
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for (int i = 0; i < N; i++)
    {
        W += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) * 
             Eigen::Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose(); 
    }
    // SVD on W 
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU|Eigen::ComputeFullV); 
    Eigen::Matrix3d U = svd.matrixU(); 
    Eigen::Matrix3d V = svd.matrixV(); 

    if (U.determinant() * V.determinant() < 0)
	{
        for (int x = 0; x < 3; ++x)
        {
            U(x, 2) *= -1;
        }
	}
    Eigen::Matrix3d R_ = U * (V.transpose()); 
    Eigen::Vector3d t_ = Eigen::Vector3d ( p1.x, p1.y, p1.z ) - 
                         R_ * Eigen::Vector3d ( p2.x, p2.y, p2.z );
    // convert to cv::Mat
    auto rot = Eigen::AngleAxisd(R_).axis();
    R[0] = rot[0]; 
    R[1] = rot[1]; 
    R[2] = rot[2]; 

    

    t[0] = t_(0,0); 
    t[1] = t_(1,0); 
    t[2] = t_(2,0); 
    

}

int main( int argc, char** argv )
{
    std::cout<<"Hello 591!"<<std::endl;
    
    //Read data
    ParameterReader pd;
    int display = atoi( pd.getData( "display" ).c_str() );
    int imgDisplay = atoi( pd.getData( "imageDisplay" ).c_str() );
    int width  =   atoi( pd.getData( "width" ).c_str() );
    int height    =   atoi( pd.getData( "height"   ).c_str() );
    std::string filePath1 = pd.getData("testImage1"); 
    std::string filePath2 = pd.getData("testImage2"); 

    FRAME f1 = readImage(filePath1, &pd);
    FRAME f2 = readImage(filePath2, &pd);
    std::cout << "working on : \n" << filePath1 << "\n" << filePath2 << endl;  

    if (imgDisplay)
    {
        cv::imshow("Frame1", f1.rgb); 
        cv::imshow("Frame2", f2.rgb); 
        cv::waitKey(0); 
    }    
    // Sift feature extraction
    // sift feature is in the xfeatures2d

    int nfeatures = atoi( pd.getData( "nfeatures" ).c_str() );
    int nOctaveLayers =  atoi( pd.getData( "nOctaveLayers" ).c_str() );
    double contrastThreshold = atoi(pd.getData( "contrastThreshold" ).c_str())/100.0; 
    double edgeThreshold  = atoi(pd.getData( "edgeThreshold" ).c_str()) * 1.0; 
    double sigma         = atoi(pd.getData( "sigma" ).c_str())/10.0; 

    float scaleFactor = atoi(pd.getData( "scaleFactor" ).c_str())/100.0; 
    if (display)
    {
        cout << "Image height " << height << " width is " << width << endl; 
        cout << "nFeature: " << nfeatures << endl << "nOctaveLayers : " << nOctaveLayers << endl;
        cout << "scaleFactor: " << scaleFactor << endl; 
        cout << "nOctaveLayers: " << nOctaveLayers << endl; 
    }
    //cv::Ptr<cv::Feature2D> detector = 
    //        cv::xfeatures2d::SIFT::create(nfeatures,nOctaveLayers, contrastThreshold,sigma);
    //float     
    //cv::Ptr<cv::DescriptorExtractor> descriptor; 
    auto detector = cv::ORB::create(nfeatures, scaleFactor, nOctaveLayers, 31);
    
    detector->detectAndCompute(f1.rgb, cv::Mat(), f1.kp, f1.desp);

    detector->detectAndCompute(f2.rgb, cv::Mat(), f2.kp, f2.desp); 

    // find matches
    std::vector<cv::DMatch> matches; 
    // flann mather
    cv::Ptr<cv::DescriptorMatcher> matcher = 
        cv::makePtr<cv::FlannBasedMatcher>(cv::makePtr<cv::flann::LshIndexParams>(12,20, 2));
    //cv::BFMatcher matcher(cv::NORM_HAMMING, true); 
    //cv::FlannBasedMatcher matcher; 
    matcher->match(f1.desp, f2.desp, matches); 
    
    /*
    cout<<"Find total "<<matches.size()<<" matches."<<endl;
    cv::Mat imgMatches;
    cv::drawMatches( f1.rgb, f1.kp, f2.rgb, f2.kp, matches, imgMatches );
    cv::imshow( "matches", imgMatches );
    cv::imwrite( "matches.png", imgMatches );
    cv::waitKey( 0 );
    */
    std::vector<cv::DMatch> goodMatches; 
    double minDis = 999;
    // get the smallest dist 
    for (size_t i = 0; i < matches.size(); ++i)
    {
        if ( matches[i].distance < minDis )
            minDis = matches[i].distance;
    }
    minDis += 0.000001; 
    //cout<<"min dis = "<<minDis<<endl;
    // get the good matches
    int scaleOfGoodMatch = atoi( pd.getData( "scaleOfGoodMatch" ).c_str() );
    for ( size_t i=0; i<matches.size(); i++ )
    {
        if (matches[i].distance <= scaleOfGoodMatch*minDis)
            goodMatches.push_back( matches[i] );
    }

    // display good matches
    if (display)
    {
        cout<<"Find total "<<matches.size()<<" matches."<<endl;
        cout<<"good matches="<<goodMatches.size()<<endl;
        
    }
    if (imgDisplay)
    {
        cv::Mat imgMatches;
        cv::drawMatches( f1.rgb, f1.kp, f2.rgb, f2.kp, goodMatches, imgMatches );
        cv::imwrite( "good_matches.png", imgMatches );
        cv::imshow( "good matches", imgMatches );
        cv::waitKey(0); 
    }


    // 3D poitns
    std::vector<cv::Point3d> src ; 
    std::vector<cv::Point3d> dst; 
    // 2D location
    std::vector<cv::Point2d> imagCor; 

    for (size_t i = 0; i<goodMatches.size(); ++i)
    {

        
        cv::Point2d p1 = f1.kp[goodMatches[i].queryIdx].pt;
        cv::Point2d p2 = f2.kp[goodMatches[i].trainIdx].pt;
        
        cv::Point3d point1; 
        cv::Point3d point2;
        //cout << p1.x << " " << p2.x << endl; 
        //cout << p1.y << " " << p2.y << endl; 
        //cout << endl;  

        point1.x = f1.depth_x.at<double>(int(p1.y), int(p1.x)); 
        point1.y = f1.depth_y.at<double>(int(p1.y), int(p1.x)); 
        point1.z = f1.depth_z.at<double>(int(p1.y), int(p1.x));
        point2.x = f2.depth_x.at<double>(int(p2.y), int(p2.x)); 
        point2.y = f2.depth_y.at<double>(int(p2.y), int(p2.x)); 
        point2.z = f2.depth_z.at<double>(int(p2.y), int(p2.x));

        /*
        point1.x = f1.depth.at<double>(int(p1.y), int(p1.x), 0); 
        point1.y = f1.depth.at<double>(int(p1.y), int(p1.x), 1); 
        point1.z = f1.depth.at<double>(int(p1.y), int(p1.x), 2);
        point2.x = f2.depth.at<double>(int(p2.y), int(p2.x), 0); 
        point2.y = f2.depth.at<double>(int(p2.y), int(p2.x), 1); 
        point2.z = f2.depth.at<double>(int(p2.y), int(p2.x), 2);
        */
        if (display)
        {
            //cout << point1 << endl; 
            //cout << point2 << endl; 
            //cout << endl; 
        }
        src.push_back(point1); 
        dst.push_back(point2);
        //imagCor.push_back(cv::Point2f(f2.kp[goodMatches[i].trainIdx].pt)); 
    }
    
    cv::Mat rvec, translationVec, inliers, ratationVector;
    cv::Mat affine = cv::Mat::zeros(3,4,CV_64F);
    if (display)
    {
        cout<<"src.size "<<src.size()<<endl;
        cout<<"dst.size "<<dst.size()<<endl;
    }
    
    int half = src.size() * 0.6;
    double threshold = 0.0; 
    int count = 0; 

    while (count < half)
    {
        threshold += 0.2;
        cv::estimateAffine3D(src, dst,affine,inliers, threshold ,0.99999);
        count = 0; 
        for (int i = 0; i < src.size(); ++i)
        {
            if(inliers.at<bool>(0,i) == true)
            {
                ++count; 
            }
        }
    }

    std::cout << "Inliners : " << count << " Total : " << src.size() << std::endl;
    std::cout << "thres hold " << threshold << std::endl;  

    //std::cout << inliers << std::endl; 
    //std::cout << src.size() << std::endl; 
    int writeImg = atoi( pd.getData( "writeImg" ).c_str() );
    if (writeImg)
    {
        cv::Mat imgMatches;
        std::vector<cv::DMatch> goodMatches2;
        for (int i = 0; i < src.size(); ++i)
        {
            //std::cout << inliers.at<bool>(0,i) << std::endl; 
            if(inliers.at<bool>(0,i) == true)
            {
                goodMatches2.push_back(goodMatches[i]); 
            }
        }
        cv::drawMatches( f1.rgb, f1.kp, f2.rgb, f2.kp, goodMatches2, imgMatches );
        cv::imwrite( "good_matches.png", imgMatches );
    }
    //std::cout<<"\naffine transforation is : \n"<<affine<<endl;
    
    cv::Mat ratationMatrix = affine(cv::Rect(0,0,3,3));
    cv::Rodrigues(ratationMatrix,ratationVector);

    cv::Mat R = ratationMatrix; 
    double sy= std::sqrt(R.at<double>(0,0) * R.at<double>(0,0) +  R.at<double>(1,0) * R.at<double>(1,0) );
    double roll = std::atan2(R.at<double>(2,1) , R.at<double>(2,2));
    double pitch = std::atan2(-R.at<double>(2,0), sy);
    double yaw = std::atan2(R.at<double>(1,0), R.at<double>(0,0));
    std::cout << "Roll " << roll << " pitch " << pitch << " yaw " << yaw << std::endl; 


    translationVec = affine(cv::Rect(3,0,1,3));
    ratationVector = ratationVector * (180 / 3.14); 
    std::cout<<"\nRotation Vector :\n "<<ratationVector<<endl;
    std::cout<<"\ntranslation : \n"<<translationVec<<endl;

    std::vector<double> t(3); 
    std::vector<double> Rot(3); 

    poseEstimation3D3D(src, dst, Rot, t); 
    cv::Rodrigues(R,ratationVector);
    std::cout << " \n3D3D SVD Result" << std::endl; 
    std::cout << Rot[0] << " " << Rot[1] << " " << Rot[2] << std::endl; 
    std::cout << " \ntranslation Result \n" << t[0] << " " << t[1] << " " << t[2] << std::endl; 

   



    return 0; 

}