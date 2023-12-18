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
      <div>
        <div style={{ marginTop: "30px" }}>
          <span style={{ marginRight: "10px" }}>Select an image file:</span>
          <input
            type="file"
            name="file"
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
            <div className="image-card">
              <div style={{ margin: "10px" }}>↓↓↓ The original image ↓↓↓</div>
              <img
                alt="Original input"
                src={imgUrl}
                onLoad={(e) => {
                  this.processImage(e.target);
                }}
              />
            </div>

            <div className="image-card">
              <div style={{ margin: "10px" }}>↓↓↓ The gray scale image ↓↓↓</div>
              <canvas ref={this.grayImgRef} />
            </div>

            <div className="image-card">
              <div style={{ margin: "10px" }}>↓↓↓ Canny Edge Result ↓↓↓</div>
              <canvas ref={this.cannyEdgeRef} />
            </div>

            <div className="image-card">
              <div style={{ margin: "10px" }}>↓↓↓ Gaussia Blur ↓↓↓</div>
              <canvas ref={this.gausBlurRef} />
            </div>

            <div className="image-card">
              <div style={{ margin: "10px" }}>↓↓↓ Median Blur ↓↓↓</div>
              <canvas ref={this.medianBlurRef} />
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
