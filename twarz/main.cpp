#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xobjdetect.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>
#include <algorithm>
#include <string>
#include <opencv2/imgproc/bindings.hpp>
#include <opencv2/features2d.hpp>


using namespace std;
string model_pipe = "face_mesh_Nx3x192x192.onnx";
cv::Mat prostokat_twarzy; // !! giga wazna zmienna przechowuje punkty twaarzy dzieki ktorym powstaje elipsa
cv::Mat ramka,zdjecie,twarz,wyciecie_twarzy,obszar_twarzy;
string sciezka_yunet = "face_detection_yunet_2023mar.onnx";


cv::Ptr<cv::FaceDetectorYN> wykrywacz= cv::FaceDetectorYN::create(sciezka_yunet,"",cv::Size(1920,1080), 0.7 , 0.3 ,5000);
int main() {
    cv::VideoCapture kamera(0); //jak mamy 2 kamery to pobawic sie indeksami
    while (kamera.isOpened()) {
        // zmienna typu macierz xD
        kamera >> ramka;
         //haar od szukania twarzy
        wykrywacz->detect(ramka,prostokat_twarzy);
        for (int i = 0; i < prostokat_twarzy.rows; i++) {
            int x1 = (int)prostokat_twarzy.at<float>(i,0) ;
            int y1 = (int)(prostokat_twarzy.at<float>(i,1) );
            int w1 = (int)prostokat_twarzy.at<float>(i,2);
            int h1 = (int)prostokat_twarzy.at<float>(i,3);

            int x = (x1 - 100);
            int y = (y1 - 250);
            int w = (w1 + 200);
            int h = (h1 + 300);


    cv::Point srodek(x+w/2, y+h/2);
            cv::Point os_elipsy(w/2, h/2);


            cv::Mat podglad = ramka.clone();
            cv::ellipse(podglad, srodek, os_elipsy, 0, 0, 360, cv::Scalar(255), 3);
            cv::imshow("zdjecie", podglad);




            if (cv::waitKey(1) == 32)
            {
                cv::imwrite("zdjecie.jpg", ramka);
                cv::Rect obszar_twarz(srodek.x - 200, srodek.y - 400, w+70, h+90 );

                wyciecie_twarzy = ramka(obszar_twarz).clone();

                cv::imwrite("wyciecie_twarzy.jpg", wyciecie_twarzy);;


                break;
            }
            if (cv::waitKey(1) == 32)
                break;
        }
        if (cv::waitKey(1) == 32)
            break;
    }







    cv::Mat resized;
    cv::resize(wyciecie_twarzy, resized, cv::Size(600, 600), 0, 0, cv::INTER_AREA);
    cv::Mat gray;
    cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);
    cv::Mat filtered;
    cv::bilateralFilter(gray, filtered, 9, 75, 75);
    auto clahe = cv::createCLAHE(2.5, cv::Size(8, 8));
    cv::Mat enhanced;
    clahe->apply(filtered, enhanced);
    cv::Mat edges;
    cv::Canny(enhanced, edges, 40, 110);
    vector<vector<cv::Point>> contours;
    cv::findContours(edges, contours,    cv::RETR_LIST, cv::CHAIN_APPROX_TC89_KCOS);
    cv::Mat szkic = cv::Mat::zeros(resized.size(), CV_8UC3);
    ofstream plik("punkty_robota.csv");
    plik << "sciezka,nr,x,y\n";
    int nr = 0;
    for (size_t i = 0; i < contours.size(); i++) {
        double len = cv::arcLength(contours[i], false);
        if (len <= 14.0)
            continue;
        vector<cv::Point> approx;
        cv::approxPolyDP(contours[i], approx, 0.0035 * len, false);
        if (approx.size() < 2)
            continue;
        for (size_t k = 1; k < approx.size(); k++)
            cv::line(szkic, approx[k - 1], approx[k], cv::Scalar(240, 240, 240), 1, cv::LINE_AA);
        for (size_t k = 0; k < approx.size(); k++)
            plik << nr << "," << k << "," << approx[k].x << "," << approx[k].y << "\n";
        nr++;
    }
    plik.close();
    cv::imwrite("szkic.jpg", szkic);
    cv::imshow("szkic", szkic);
    cv::waitKey(0);
    return 0;


    /*/ Definicje indeksów punktów MediaPipe dla poszczególnych części twarzy
    const std::vector<int> OVAL = {10,  338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58,  132, 93,  234, 127, 162, 21,  54,  103, 67, 109};
    const std::vector<int> LIPS_OUTER = {61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185};
    const std::vector<int> LIPS_INNER = {78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191};
    const std::vector<int> LEFT_EYE = {263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466};
    const std::vector<int> RIGHT_EYE = {33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246};
    const std::vector<int> LEFT_EYEBROW = {276, 283, 282, 295, 285, 300, 293, 334, 296, 336};
    const std::vector<int> RIGHT_EYEBROW = {46, 53, 52, 65, 55, 70, 63, 105, 66, 107};
    const std::vector<int> NOSE = {168, 6, 197, 195, 5, 4, 1, 2, 98, 327};
*/



    }
