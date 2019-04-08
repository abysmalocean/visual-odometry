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

/*
// FRAME Struct
struct FRAME
{
    int frameID; 
    cv::Mat rgb, depth; // image and depth
    cv::Mat desp;       // descriptor
    vector<cv::KeyPoint> kp; // key points
};
*/
using std::cout; 
using std::endl;

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
    cv::Mat imgMatches;
    cv::drawMatches( f1.rgb, f1.kp, f2.rgb, f2.kp, goodMatches, imgMatches );
    cv::imwrite( "good_matches.png", imgMatches );
    if (imgDisplay)
    {
        cv::imshow( "good matches", imgMatches );
        cv::waitKey(0); 
    }


    // 3D poitns
    std::vector<cv::Point3f> src ; 
    std::vector<cv::Point3f> dst; 
    // 2D location
    std::vector<cv::Point2f> imagCor; 

    for (size_t i = 0; i<goodMatches.size(); ++i)
    {

        
        cv::Point2f p1 = f1.kp[goodMatches[i].queryIdx].pt;
        cv::Point2f p2 = f2.kp[goodMatches[i].trainIdx].pt;
        
        cv::Point3f point1; 
        cv::Point3f point2;
        //cout << p1.x << " " << p2.x << endl; 
        //cout << p1.y << " " << p2.y << endl; 
        //cout << endl;  

        point1.x = f1.depth.at<double>(int(p1.x), int(p1.y), 0); 
        point1.y = f1.depth.at<double>(int(p1.x), int(p1.y), 1); 
        point1.z = f1.depth.at<double>(int(p1.x), int(p1.y), 2);
        
        point2.x = f2.depth.at<double>(int(p2.x), int(p2.y), 0); 
        point2.y = f2.depth.at<double>(int(p2.x), int(p2.y), 1); 
        point2.z = f2.depth.at<double>(int(p2.x), int(p2.y), 2);
        if (display)
        {
            cout << point1 << endl; 
            cout << point2 << endl; 
            cout << endl; 
        }
        src.push_back(point1); 
        dst.push_back(point2);
        //imagCor.push_back(cv::Point2f(f2.kp[goodMatches[i].trainIdx].pt)); 
    }
    
    cv::Mat rvec, tvec, inliers, rvecN;
    cv::Mat outM3by4 = cv::Mat::zeros(3,4,CV_64F);
    if (display)
    {
        cout<<"src.size "<<src.size()<<endl;
        cout<<"dst.size "<<dst.size()<<endl;
    }
    cv::estimateAffine3D(src, dst,outM3by4,inliers,3,0.9999);

    cout<<"affine M = "<<outM3by4<<endl;
    
    cv::Mat rmat = outM3by4(cv::Rect(0,0,3,3));
    cv::Rodrigues(rmat,rvecN);

    cv::Mat tvecN = outM3by4(cv::Rect(3,0,1,3));
    cout<<"R="<<rvecN<<endl;
    cout<<"t="<<tvecN<<endl;
    cout<<"inliers: "<<inliers.rows<<endl; 


    return 0; 

}