
#include <../include/utility.h>
#include <string>
#include <iostream>
#include <omp.h>

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

// read SwissRanger SR4000
FRAME readImage(std::string FileName, ParameterReader *pd, int ID)
{
    // std::cout<< "in image read file " << std::endl; 

    int width  =   atoi( pd->getData( "width" ).c_str() );
    int height    =   atoi( pd->getData( "height"   ).c_str() );
    int display = atoi( pd->getData( "display" ).c_str() );
    int imageDisplay = atoi( pd->getData( "imageDisplay" ).c_str() );

    FRAME f;
    //cv::Mat gray  = cv::Mat::zeros(height,width, CV_64F); 
    cv::Mat gray  = cv::Mat::zeros(height, width, CV_64F); 
    int size[3] = {height, width, 3}; 
    cv::Mat depth(3, size , CV_64F, cv::Scalar(0));
    std::cout << "Size is " << depth.size() << std::endl; 

    std::ifstream imageFile(FileName); 
    std::string str; 
    std::string tmp; 
    size_t lineCount = 0; 

    while(getline(imageFile, str))
    {
        // 
        // every line is 176 data, for each row
        // get the first depth, channel 2
        ++lineCount;
        if (lineCount > 0 && lineCount < height + 1 ) 
        {
            for (int i = 0; i < width; ++i)
            {
                imageFile >> depth.at<double>(lineCount-1,i,2); 
            }
        }
        // get the second depth, channel 0 
        else if(lineCount > (height + 1) && lineCount < (height + 1)* 2)
        {
            for (int i = 0; i < width; ++i)
            {
                imageFile >> depth.at<double>(lineCount - height, i, 0); 
            }
        }

        // get the third depth, channel 1 
        else if(lineCount > (height + 1) * 2 && lineCount < (height + 1) * 3)
        {
            for (int i = 0; i < width; ++i)
            {
                imageFile >> depth.at<double>(lineCount - 1- 2 * (height + 1), i, 1); 
            }
        }

        // get the gray image
        else if (lineCount > (height + 1)* 3 && lineCount < (height + 1) * 4)
        {
            for (int i = 0; i < width; ++i)
            {
                imageFile >> gray.at<double>(lineCount - 1 - 3 * (height+1), i, 0);
                //std::cout << gray.at<double>(lineCount - 3*height, i, 0); 
            }
            //std::cout <<std::endl;
        }
    }  
    

    //cv::normalize(gray, gray, 1.0, 0.0, cv::NORM_L2); 
     
    cv::Mat means, stddev;
	cv::meanStdDev(gray, means, stddev);
    if(display)
    {
        printf("mean: %.2f, stddev: %.2f\n", means.at<double>(0), stddev.at<double>(0));
    }
    //gray.convertTo(gray, CV_32F);
    //cv::medianBlur(gray, gray, 5);
    //cv::GaussianBlur(gray, gray, cv::Size(3,3),1.0,1.0,4);
    // filter out the extream large value, this value is error for some of the reason
    #pragma omp parallel for
    for (int j = 0; j < height; ++j)
    {
        for (int i = 0; i < width; ++i)
        {
            if (gray.at<double>(j, i, 0) > means.at<double>(0) + 4 * stddev.at<double>(0)) 
            {
                gray.at<double>(j, i, 0) =  means.at<double>(0) + 4 * stddev.at<double>(0); 
            }
            
        }
    }

    cv::normalize(gray, gray, 1.0, 0.0, cv::NORM_MINMAX);
    f.depth = depth.clone(); 

    /*
        // extract planes from matrix
    std::vector<cv::Mat> matVec;
    for (int p = 0; p < dims[2]; ++p) {
        double *ind = (double*)mnd.data + p * dims[0] * dims[1]; // sub-matrix pointer
        matVec.push_back(cv::Mat(2, dims, CV_64F, ind).clone()); // clone if mnd goes away
    }
    */
    std::vector<cv::Mat> matVec;
    
    for (int p = 0; p < size[2]; ++p) {
        double *ind = (double*)depth.data + p * size[1] * size[0]; // sub-matrix pointer
        matVec.push_back(cv::Mat(2, size, CV_64F, ind).clone()); // clone if mnd goes away
    }
    if(imageDisplay)
    {
        cv::normalize(matVec[0], matVec[0], 1.0, 0.0, cv::NORM_MINMAX);
        cv::normalize(matVec[1], matVec[1], 1.0, 0.0, cv::NORM_MINMAX);
        cv::normalize(matVec[1], matVec[1], 1.0, 0.0, cv::NORM_MINMAX);
        matVec[0] *= 255.0;
        matVec[1] *= 255.0;
        matVec[2] *= 255.0;
        matVec[0].convertTo(matVec[0],CV_8UC1); 
        matVec[1].convertTo(matVec[0],CV_8UC1); 
        matVec[2].convertTo(matVec[0],CV_8UC1); 
        cv::imshow("fist channel ", matVec[0]); 
        cv::imshow("Second Channel ", matVec[1]); 
        cv::imshow("third Channel ", matVec[2]); 
        cv::waitKey(9);
    }


    //cv::GaussianBlur(depth, f.depth, cv::Size(5,5), 0);

    gray *= 255.0; 
    gray.convertTo(gray, CV_8UC1);

    //cv::equalizeHist(gray,gray); 
    f.rgb = gray.clone(); 
    f.depth = depth.clone(); 
    return f; 
}

