import React from "react";
import cv from "@techstark/opencv-js";
import { loadHaarFaceModels, detectHaarFace } from "./haarFaceDetection";

class TestPage extends React.Component {
  constructor(props) {
    super(props);
    this.inputImgRef = React.createRef();
    this.grayImgRef = React.createRef();
    this.cannyEdgeRef = React.createRef();
    this.gausBlurRef = React.createRef();
    this.medianBlurRef = React.createRef();
    this.haarFaceImgRef = React.createRef();
    this.state = {
      imgUrl: null,
    };
  }

  componentDidMount() {
    loadHaarFaceModels();
  }

  /////////////////////////////////////////
  //
  // process image with opencv.js
  //
  /////////////////////////////////////////
  async processImage(imgSrc) {
    const img = cv.imread(imgSrc);

    // to gray scale
    const imgGray = new cv.Mat();
    cv.cvtColor(img, imgGray, cv.COLOR_BGR2GRAY);
    cv.imshow(this.grayImgRef.current, imgGray);

    // detect edges using Canny
    const edges = new cv.Mat();
    cv.Canny(imgGray, edges, 100, 100);
    cv.imshow(this.cannyEdgeRef.current, edges);

    // Gaussian Blur
    const gaus = new cv.Mat();
    let ksize = new cv.Size(5, 5);
    let sigmaX = 10;
    cv.GaussianBlur(img, gaus, ksize, sigmaX, 0, cv.BORDER_DEFAULT);
    cv.imshow(this.gausBlurRef.current, gaus);

    // Median Blur
    const median = new cv.Mat();
    let ksize3 = 7;
    cv.medianBlur(img, median, ksize3);
    cv.imshow(this.medianBlurRef.current, median);

    // detect faces using Haar-cascade Detection
    const haarFaces = await detectHaarFace(img);
    cv.imshow(this.haarFaceImgRef.current, haarFaces);

    // need to release them manually
    img.delete();
    imgGray.delete();
    edges.delete();
    haarFaces.delete();
  }

  render() {
    const { imgUrl } = this.state;
    return (
      <div className="w-[1152px] max-w-6xl">
        <h1 className="text-6xl py-12 font-extrabold text-center">
          Signals and System Analysis in Image Processing
        </h1>
        <div className="pb-12 flex items-center justify-center gap-4">
          <span className="text-xl">Select an image file:</span>
          <input
            type="file"
            name="file"
            className="block text-sm text-slate-500
            file:mr-4 file:py-2 file:px-4
            file:rounded-full file:border-0
            file:text-sm file:font-semibold
            file:bg-violet-50 file:text-violet-700
            hover:file:bg-violet-100
      "
            accept="image/*"
            onChange={(e) => {
              if (e.target.files[0]) {
                this.setState({
                  imgUrl: URL.createObjectURL(e.target.files[0]),
                });
              }
            }}
          />
        </div>
        {imgUrl && (
          <div className="images-container">
            <div className="image-card shadow-md overflow-hidden rounded-md">
              <img
                alt="Original input"
                src={imgUrl}
                onLoad={(e) => {
                  this.processImage(e.target);
                }}
              />
              <div className="text-xl py-4 text-center">Original Image</div>
            </div>

            <div className="image-card shadow-md overflow-hidden rounded-md">
              <canvas ref={this.grayImgRef} />
              <div className="text-xl py-4 text-center">Gray Scale</div>
            </div>

            <div className="image-card shadow-md overflow-hidden rounded-md">
              <canvas ref={this.cannyEdgeRef} />
              <div className="text-xl py-4 text-center">
                Canny Edge Detection
              </div>
            </div>

            <div className="image-card shadow-md overflow-hidden rounded-md">
              <canvas ref={this.gausBlurRef} />
              <div className="text-xl py-4 text-center">Gaussian Blur</div>
            </div>

            <div className="image-card shadow-md overflow-hidden rounded-md">
              <canvas ref={this.medianBlurRef} />
              <div className="text-xl py-4 text-center">Median Blur</div>
            </div>

            {/* <div className="image-card">
              <div style={{ margin: "10px" }}>
                ↓↓↓ Haar-cascade Face Detection Result ↓↓↓
              </div>
              <canvas ref={this.haarFaceImgRef} />
            </div> */}
          </div>
        )}
      </div>
    );
  }
}

export default TestPage;
