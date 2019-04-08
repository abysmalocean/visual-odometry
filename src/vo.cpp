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
#include <dirent.h>
#include <omp.h>

using std::cout; 
using std::endl;

int main( int argc, char** argv )
{
    std::cout<<"Hello 591! First part of the project!!"<<std::endl;
    
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

    int nfeatures = atoi( pd.getData( "nfeatures" ).c_str() );
    int nOctaveLayers =  atoi( pd.getData( "nOctaveLayers" ).c_str() );
    double contrastThreshold = atoi(pd.getData( "contrastThreshold" ).c_str())/100.0; 
    double edgeThreshold  = atoi(pd.getData( "edgeThreshold" ).c_str()) * 1.0; 
    double sigma         = atoi(pd.getData( "sigma" ).c_str())/10.0; 
    float scaleFactor = atoi(pd.getData( "scaleFactor" ).c_str())/100.0; 
    auto detector = cv::ORB::create(nfeatures, scaleFactor, nOctaveLayers, 31);
    // flann mather
    cv::Ptr<cv::DescriptorMatcher> matcher = 
        cv::makePtr<cv::FlannBasedMatcher>(cv::makePtr<cv::flann::LshIndexParams>(12,20, 2));
    
    std::vector<FRAME> source; 
    std::vector<FRAME> distination;

    std::string folder1 = pd.getData("folder1"); 
    std::string folder2 = pd.getData("folder2"); 
    size_t count = 0; 
    if (auto dir = opendir(folder1.c_str())) 
    {
        while (auto f = readdir(dir)) 
        {
            if (!f->d_name || f->d_name[0] == '.')
                continue; // Skip everything that starts with a dot
            std::string filePath = folder1 + f->d_name; 
            if(display) {printf("File: %s | num [%zu]\n", f->d_name, count);}
            FRAME frame = readImage(filePath, &pd);
            frame.frameID = count++; 
            source.push_back(frame); 
        }
        closedir(dir);
    }
    printf("Source finish \n"); 
    if (auto dir = opendir(folder2.c_str())) 
    {
        while (auto f = readdir(dir)) 
        {
            if (!f->d_name || f->d_name[0] == '.')
                continue; // Skip everything that starts with a dot
            std::string filePath = folder2 + f->d_name; 
            if(display) {printf("File: %s | num [%zu]\n", f->d_name, count);}
            FRAME frame = readImage(filePath, &pd);
            frame.frameID = count++; 
            distination.push_back(frame); 
        }
        closedir(dir);
    }
    printf("finsih loading the data\n"); 
    #pragma omp parallel for
    for (int i = 0; i < source.size(); ++i)
    {
        detector->detectAndCompute(source[i].rgb, cv::Mat(), 
                                    source[i].kp, source[i].desp);
    }

    #pragma omp parallel for
    for (int i = 0; i < distination.size(); ++i)
    {
        detector->detectAndCompute(distination[i].rgb, cv::Mat(), 
                                    distination[i].kp, distination[i].desp);
    }
    printf("Source finish calcaulte the descriptor\n"); 
    std::vector<cv::Mat> R(source.size()*distination.size()); 
    std::vector<cv::Mat> T(source.size()*distination.size()); 



    for (int i = 0; i < source.size(); ++i)
    {
            printf("working on [%d]\n", i); 
        
        for (int j = 0; j < distination.size(); ++j)
        {
            f1 = source[i]; 
            f2 = distination[j]; 
            std::vector<cv::DMatch> matches; 
            matcher->match(source[i].desp, distination[j].desp, matches);

            std::vector<cv::DMatch> goodMatches; 
            double minDis = 999;
            // get the smallest dist 
            for (size_t i = 0; i < matches.size(); ++i)
            {
                if ( matches[i].distance < minDis )
                    minDis = matches[i].distance;
            }
            minDis += 0.000001; 
            int scaleOfGoodMatch = atoi( pd.getData( "scaleOfGoodMatch" ).c_str() );
            for ( size_t i=0; i<matches.size(); i++ )
            {
                if (matches[i].distance <= scaleOfGoodMatch*minDis)
                    goodMatches.push_back( matches[i] );
            }
            // 3D poitns
            std::vector<cv::Point3f> src; 
            std::vector<cv::Point3f> dst; 
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
                src.push_back(point1); 
                dst.push_back(point2);
            }
            cv::Mat rvec, tvec, inliers, rvecN;
            cv::Mat outM3by4 = cv::Mat::zeros(3,4,CV_64F);
            cv::estimateAffine3D(src, dst,outM3by4,inliers,3,0.9999);
            cv::Mat rmat = outM3by4(cv::Rect(0,0,3,3));
            cv::Rodrigues(rmat,rvecN);
            cv::Mat tvecN = outM3by4(cv::Rect(3,0,1,3));

            R[i*source.size() + j] = rvecN;
            T[i*source.size() + j] = tvecN; 

            if (display)
            {
                cout<<"R="<<rvecN<<endl;
                cout<<"t="<<tvecN<<endl;
            }

        }
        
        for (int i = 0; i < source.size()*distination.size(); ++i)
        {

        }

        
    }

    return 0; 

}