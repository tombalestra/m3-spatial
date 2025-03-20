visualizer_prompt = '''
I want to build a web application for gassuan splatting visualization both in rgb and feature space.
For the basic architecture: - Frontend: HTML, JavaScript (WebGL) - Backend: Python (torch cuda)
Here is the requirement:
1. The front end should be able to accept keyboard up/down/left/right to adjust the camera pose.
2. The front end should be able to use mouse for drag to rotate.
3. The window could be embed into an HTML page for visualization.
4. The backend should be able to generate RGB image and feature PCA for visualization given camera view.
For simplicity, the backend now could only have a function that using a random matric to map the camear pose to noise image with size 256x256.
Attached is an example of the result I want to achieve, please generate codes for html, js and python.
'''